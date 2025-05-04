# Implementation of Reinforcement Learning agents that interact with the
# educational domain environment implemented in Racket.

import os
import argparse
import collections
import copy
from dataclasses import dataclass
import time
from datetime import datetime
import itertools
import random
import json
import pickle
import math
import subprocess
import logging
import numpy as np
import traceback

import gc

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import wandb

import util
from util import register
from environment import Environment, State, Action
from evaluation import EnvironmentWithEvaluationProxy, evaluate_policy, evaluate_policy_checkpoints, EndOfLearning
from q_function import QFunction, InverseLength, RandomQFunction, RubiksGreedyHeuristic

import steps
from compress import COMPRESSORS
from abstractions import Axiom, ABS_TYPES

import tqdm
# import time

SUCCESS_STATE = State(['success'], [], 1.0)
AXIOMS = {
    "equations-ct": ["refl", "comm", "assoc", "dist", "sub_comm", "eval", "add0", "sub0", "mul1", "div1",
                     "div_self", "sub_self", "subsub", "mul0", "zero_div", "add", "sub", "mul", "div"],
    "equations-hard": ["refl", "comm", "assoc", "dist", "sub_comm", "eval", "add0", "sub0", "mul1", "div1",
                       "div_self", "sub_self", "subsub", "mul0", "zero_div", "add", "sub", "mul", "div"],
    "fractions": ["mul", "combine", "mfrac", "simpl1", "scale", "cancel", "eval", "factorize"],
    "fractions-hard": ["mul", "combine", "mfrac", "simpl1", "scale", "cancel", "eval", "factorize"]
}

class LearningAgent:
    '''Algorithm that guides learning via interaction with the enviroment.
    Gets to decide when to start a new problem, what states to expand, when to take
    random actions, etc.

    Any learning algorithm can be combined with any Q-Function.
    '''

    subtypes: dict = {}

    def learn_from_environment(self, environment):
        "Lets the agent learn by interaction using any algorithm."
        raise NotImplementedError()

    def learn_from_experience(self):
        "Lets the agent optionally learn from its past interactions one last time before eval."

    def stats(self):
        "Returns a string with learning statistics for this agent, for debugging."
        return ""

    def get_q_function(self):
        "Returns a QFunction that encodes the current learned model."
        raise NotImplementedError()

    @staticmethod
    def new(q_fn, config):
        return LearningAgent.subtypes[config['type']](q_fn, config)


@dataclass
class ContrastiveExample:
    "Keeps track of one contrastive example (one positive vs N negative actions)"
    positive: Action
    negatives: list[Action]
    gap: int  # How many steps into the future is this example for.

@register(LearningAgent)
class NCE(LearningAgent):
    "Agent that uses the InfoNCE contrastive loss to differentiate positive/negative actions"
    def __init__(self, q_function, config):
        self.q_function = q_function
        replay_buffer_size = config.get('replay_buffer_size', 10**6) # Default size if not specified

        self.examples = collections.deque(maxlen=replay_buffer_size)

        # Global state buffer for enhanced negative sampling (Seems OK - uses maxlen and stores strings)
        self.use_global_buffer = config.get('use_global_buffer', False)
        global_buffer_size = config.get('global_buffer_size', 10**6)
        self.global_state_buffer = collections.deque(maxlen=global_buffer_size) if self.use_global_buffer else None

        # Configuration for embedding-based negative sampling (parameters, likely OK)
        self.use_embedding_based_sampling = config.get('use_embedding_based_sampling', False)
        self.num_candidate_negatives = config.get('num_candidate_negatives', 1000)
        self.num_final_negatives = config.get('num_final_negatives', 128)
        self.similarity_to_current = config.get('similarity_to_current', False)

        ex_sol = config.get('example_solutions')
        self.example_solutions = None
        if isinstance(ex_sol, str):
            with open(ex_sol, 'rb') as f:
                self.example_solutions = pickle.load(f)  # tuple of Solution objects to learn from
        elif isinstance(ex_sol, (tuple, list)) and all(isinstance(sol, steps.Solution) for sol in ex_sol):
            self.example_solutions = ex_sol

        num_store_sol = config.get('num_store_sol')
        self.stored_solutions = None if num_store_sol is None else collections.deque(maxlen=num_store_sol)

        self.training_problems_explored = 0
        self.training_problems_solved = 0
        self.training_acc_moving_average = 0.0

        self.max_depth = config['max_depth']
        self.depth_step = config.get('depth_step')
        self.initial_depth = config.get('initial_depth')
        self.step_every = config.get('step_every')
        self.beam_size = config.get('beam_size')
        self.epsilon = config.get('epsilon', 0.0)

        self.optimize_every = config.get('optimize_every')
        self.gd_evaluate_every = config['gd_evaluate_every'] if self.optimize_every is None else None
        self.n_gradient_steps = config.get('n_gradient_steps', 64)
        self.beam_negatives_frac = config.get('beam_negatives_frac', 1.0)

        self.bootstrapping = config['q_function'].get('load_pretrained') is None and self.example_solutions is None
        if self.bootstrapping:
            bootstrap_from = config.get('bootstrap_from', 'Random')

            if bootstrap_from == 'InverseLength':
                self.bootstrap_policy = InverseLength(self.q_function.device)
            elif bootstrap_from == 'RubiksGreedyHeuristic':
                self.bootstrap_policy = RubiksGreedyHeuristic(self.q_function.device)
            else:
                self.bootstrap_policy = RandomQFunction(self.q_function.device)

            self.n_bootstrap_problems = config.get('n_bootstrap_problems', 100)

        self.add_success_state = config.get('add_success_state', False)
        self.keep_optimizer = config.get('keep_optimizer', True)
        self.n_future_states = config.get('n_future_states', 1)
        self.max_negatives = config.get('max_negatives', float('inf'))
        self.learning_rate = config.get('lr', 1e-4)
        self.reset_optimizer()

        self.current_depth = self.initial_depth

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=self.learning_rate)

    def name(self):
        return 'NCE'

    def add_to_global_buffer(self, states):
        """Add states to the global state buffer"""
        if not self.use_global_buffer:
            return

        for state in states:
            # Store just the string representation to save memory
            self.global_state_buffer.append(state.facts[-1])

    def get_embedding_based_negatives(self, positive_action):
        """
        Sample negatives from global buffer based on embedding similarity

        Args:
            positive_action: The positive action for which we need negatives

        Returns:
            List of negative actions
        """
        if not self.use_global_buffer or not self.use_embedding_based_sampling or not positive_action:
             return [] # Added check for positive_action not being None

        # Get the reference state for similarity (current or next)
        if self.similarity_to_current:
            reference_state = positive_action.state
        else:
            reference_state = positive_action.next_state

        # Handle case where reference_state might be None
        if reference_state is None:
            return []

        # Sample candidate negatives similar to reference state from the global buffer
        if len(self.global_state_buffer) <= self.num_candidate_negatives:
            candidate_states_str = list(self.global_state_buffer)
        else:
            # Ensure sample size doesn't exceed buffer size
            k = min(len(self.global_state_buffer), self.num_candidate_negatives)
            candidate_states_str = random.sample(list(self.global_state_buffer), k)

        # Skip if we don't have enough candidates
        if not candidate_states_str:
            return []

        # Create state objects for the candidates
        candidate_state_objs = [State([s], [''], 0.0) for s in candidate_states_str]

        # Get embeddings from the q_function's encoder
        try:
            with torch.no_grad():
                reference_embedding = self.q_function.embed_states([reference_state])
                candidate_embeddings = self.q_function.embed_states(candidate_state_objs)

                # Compute cosine similarity
                reference_embedding = F.normalize(reference_embedding, p=2, dim=1)
                candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
                similarities = torch.mm(reference_embedding, candidate_embeddings.t()).squeeze(0)

                # Get top-k most similar states
                num_to_select = min(len(similarities), self.num_final_negatives)
                if num_to_select <= 0:
                    return []

                if len(similarities) <= num_to_select:
                    # Take all if fewer than requested
                    top_indices = torch.argsort(similarities, descending=True)
                else:
                    top_indices = torch.topk(similarities, num_to_select).indices

        except Exception as e:
            logging.error(f"Error during embedding-based negative sampling: {e}")
            return []

        # Create actions for the negative states
        negative_actions = []
        for idx in top_indices:
             # Ensure index is valid
            if idx.item() < len(candidate_state_objs):
                neg_state = candidate_state_objs[idx.item()]
                # Create a dummy action that leads to this negative state
                # Ensure positive_action.state is not None
                if positive_action.state is not None:
                    neg_action = Action(
                        positive_action.state,  # Same source state
                        f"negative_{idx.item()}",  # Dummy action name
                        neg_state,  # Target state
                        0.0  # No reward
                    )
                    negative_actions.append(neg_action)
                else:
                     logging.warning("Skipping negative action creation due to None positive_action.state")
        return negative_actions


    def learn_from_environment(self, environment):
        ex_sol_left = True if self.example_solutions and self.training_problems_explored < len(self.example_solutions) else False

        wrapper = tqdm.tqdm if self.optimize_every is None else lambda x: x
        for i in wrapper(range(self.training_problems_explored, len(self.example_solutions))
                         if self.example_solutions and environment.max_steps is None
                         else itertools.count(start=self.training_problems_explored)):
            if ex_sol_left:
                # Ensure index is valid
                if i < len(self.example_solutions):
                    ex_solution = self.example_solutions[i]
                    first_state = State([ex_solution.states[0]], [''], 0.0)
                    solution = self.beam_search(first_state, environment, ex_solution)
                    if i >= len(self.example_solutions) - 1:
                        ex_sol_left = False
                        if environment.max_steps is None:
                            raise EndOfLearning()
                else:
                    logging.warning(f"Index {i} out of bounds for example_solutions (len={len(self.example_solutions)}). Skipping.")
                    ex_sol_left = False
                    if environment.max_steps is None:
                        raise EndOfLearning() 
                    continue # Skip to next iteration or generate new problem

            else:
                problem = environment.generate_new()
                solution = self.beam_search(problem, environment)

            self.training_problems_explored += 1

            if solution is not None:
                # print(i, self.get_q_function().name(), solution.facts)
                self.training_problems_solved += 1

                if self.bootstrapping and self.training_problems_solved >= self.n_bootstrap_problems:
                    self.bootstrapping = False
                    logging.info("Bootstrapping phase finished.") # Added log
                    print("Bootstrapping phase finished.")

                if self.optimize_every is not None and self.optimize_every > 0 and self.training_problems_solved % self.optimize_every == 0:
                    logging.info('Running SGD steps.')
                    print('Running SGD steps.')
                    self.gradient_steps()

            self.training_acc_moving_average = 0.95*self.training_acc_moving_average + 0.05*int(solution is not None)

            if self.step_every is not None and self.step_every > 0 and (i + 1) % self.step_every == 0:
                self.current_depth = min(self.max_depth, self.current_depth + self.depth_step)
                logging.info(f'Beam search depth increased to {self.current_depth}.')
                print(f'Beam search depth increased to {self.current_depth}.')

    def learn_from_experience(self, env):
        if self.optimize_every is None:
            self.gradient_steps(env)

    def get_q_function(self):
        if self.bootstrapping:
            return self.bootstrap_policy
        return self.q_function

    def beam_search(self, state, environment, ex_solution=None):
        '''Performs beam search in a train problem while recording particular examples
        in the replay buffer (according to the various knobs in the algorithm, see config)
        `state`: starting state
        `ex_solution`: an example solution to carry out and to get contrastive examples from'''

        beam = [(state, None)]
        solution = None
        q = self.get_q_function()
        seen = {state.facts[-1]}
        # visited_states_with_actions stores (state, action_that_led_to_state)
        visited_states_with_actions = [[(state, None)]] # Initial state has no parent action


        # Add initial state to global buffer if enabled
        if self.use_global_buffer:
            self.add_to_global_buffer([state])

        logging.info(f'Trying {state}')

        current_max_depth = self.current_depth
        if ex_solution:
            current_max_depth = len(ex_solution.states) - 1

        for i in range(current_max_depth):
            current_beam_states = []
            for item_idx, item in enumerate(beam):
                assert isinstance(item, tuple) and len(item) == 2, \
                    f"BEAM CHECK (start loop): Item {item_idx} in 'beam' is not (State, Action/None): {type(item)} {item}"
                current_beam_states.append(item[0]) # Extract the state (index 0)

            if not current_beam_states:
                logging.info(f"Beam became empty at depth {i}. Stopping search.")
                break

            try:
                results = environment.step(current_beam_states)

                validated_results = []
                valid_indices = []
                if not isinstance(results, list):
                    logging.error(f"FATAL: environment.step did not return a list. Got: {type(results)}. Stopping search.")
                    break

                if len(results) != len(current_beam_states):
                    logging.warning(f"Mismatch between input states ({len(current_beam_states)}) and results ({len(results)}) from environment.step.")
                    min_len = min(len(results), len(current_beam_states))
                    results = results[:min_len]
                    current_beam_states = current_beam_states[:min_len] # Ensure current_beam_states aligns

                for idx, item in enumerate(results):
                    if isinstance(item, tuple) and len(item) == 2:
                        if isinstance(item[0], (int, float)) and isinstance(item[1], list):
                            validated_results.append(item)
                            valid_indices.append(idx)
                        else:
                            logging.warning(f"env.step returned invalid tuple content idx {idx}: ({type(item[0])}, {type(item[1])}). Skip.")
                    else:
                        logging.warning(f"env.step returned non-tuple/wrong size idx {idx}: {type(item)} {item}. Skip.")

                if not validated_results:
                    logging.error("No valid results from environment.step after validation. Stopping search.")
                    break

                rewards, actions_per_state = zip(*validated_results)
                valid_beam_states = [current_beam_states[i] for i in valid_indices] # States corresponding to valid results
                original_beam_valid_tuples = [beam[i] for i in valid_indices] # Keep the (State, Action) tuple from original beam

                if not valid_beam_states:
                    logging.error("No valid beam states remaining after filtering env results.")
                    break

            # --- Catch EndOfLearning specifically and re-raise ---
            except EndOfLearning:
                logging.info(f"EndOfLearning caught inside beam_search at depth {i}. Re-raising.")
                raise # Re-raise the exception to propagate it upwards

            except Exception as e:
                logging.error(f"Error during environment step or result processing at depth {i}: {e}")
                traceback.print_exc()
                break

            solution_found_this_step = False
            # Iterate using the states that produced valid results
            for beam_idx, (current_s, r) in enumerate(zip(valid_beam_states, rewards)):
                 # current_s is the state object
                 if r:
                    # Find the parent action using the original beam tuple corresponding to this valid state
                    original_parent_tuple = original_beam_valid_tuples[beam_idx]
                    # Ensure original_parent_tuple is correct format
                    assert isinstance(original_parent_tuple, tuple) and len(original_parent_tuple) == 2, \
                        f"SOLUTION CHECK: original_parent_tuple wrong format: {original_parent_tuple}"

                    parent_action = original_parent_tuple[1] # Action is the second element

                    # Double check with visited states if parent_action is None
                    if parent_action is None and i > 0:
                          logging.warning(f"Solution found for state {current_s.facts[-1]} but parent action from beam tuple is None at depth {i}. Trying visited states.")
                          # Look back in visited states using the state object current_s
                          for step_visited in reversed(visited_states_with_actions):
                                found_in_step = False
                                for visited_s, visited_pa in step_visited:
                                    if visited_pa and visited_pa.next_state == current_s:
                                        parent_action = visited_pa
                                        found_in_step = True
                                        logging.info(f"Found parent action via visited states: {parent_action.action}")
                                        break
                                if found_in_step:
                                        break


                    if parent_action is None and i == 0 and state == current_s:
                        logging.info("Problem solved at initial state.")

                    solution_state = copy.deepcopy(current_s) # Copy the state that triggered reward
                    solution_state.parent_action = parent_action # Assign the found parent action
                    solution = solution_state # Store the state object that represents the goal
                    logging.info(f"Solution found at depth {i}: {solution}")
                    solution_found_this_step = True
                    break # Exit inner solution checking loop (over valid_beam_states)


            if solution is not None: # Check the main solution variable
                # Ensure solution state is recorded for path reconstruction if needed
                if solution.parent_action: # Only add if it has a parent action
                    # Find the relevant step's visited list to append to
                    step_to_append_to = -1 # Default to last step
                    # Find the step where the parent action's state exists
                    found_parent_step = False
                    for visited_step_idx in range(len(visited_states_with_actions) -1, -1, -1):
                        for vs, va in visited_states_with_actions[visited_step_idx]:
                            if vs == solution.parent_action.state:
                                step_to_append_to = visited_step_idx + 1 # Append to the *next* step's list
                                found_parent_step = True
                                break
                        if found_parent_step: break

                    if step_to_append_to < len(visited_states_with_actions) and step_to_append_to >=0 :
                        found_in_visited = any(s_vis == solution for s_vis, _ in visited_states_with_actions[step_to_append_to])
                        if not found_in_visited:
                            visited_states_with_actions[step_to_append_to].append((solution, solution.parent_action))
                    else:
                        # Append to last list if parent step search failed (might happen if solution is found immediately)
                        found_in_visited = any(s_vis == solution for s_vis, _ in visited_states_with_actions[-1])
                        if not found_in_visited:
                            visited_states_with_actions[-1].append((solution, solution.parent_action))

                break # Exit outer loop (depth loop) because solution was found


            next_beam_candidates = [] # Stores (state, parent_action) tuples, sorted by value
            current_step_visited = [] # Stores (state, parent_action) for this step

            # If example solution is provided, follow it
            if ex_solution:
                if len(valid_beam_states) != 1 or len(actions_per_state) != 1:
                    logging.warning(f"Example Sol: Expected 1 valid state, got {len(valid_beam_states)}. Continue cautiously.")

                if valid_beam_states: # Proceed only if we have a valid state
                    target_state_fact = ex_solution.states[i+1]
                    found_next_step = False
                    current_actions = actions_per_state[0]
                    parent_state = valid_beam_states[0]

                    # Iterate over the actions to find the one leading to the target state
                    for action_idx, action in enumerate(current_actions):
                        assert isinstance(action, Action), \
                            f"EXAMPLE SOL EXP: Item {action_idx} in current_actions not Action: {type(action)}"

                        next_state = action.next_state
                        current_step_visited.append((next_state, action))
                        if next_state.facts[-1] not in seen:
                            seen.add(next_state.facts[-1])
                            if self.use_global_buffer:
                                self.add_to_global_buffer([next_state])

                        if next_state.facts[-1] == target_state_fact:
                            parent_val = parent_state.value if hasattr(parent_state, 'value') else 0.0
                            next_state.value = parent_val + 1 # Simple depth/step count value
                            # Create the tuple for the next beam
                            next_beam_candidates = [(next_state, action)]
                            found_next_step = True
                            break # Stop searching actions

                    if not found_next_step:
                        logging.error(f"Example solution step failed: Cannot find state '{target_state_fact}' from {parent_state.facts[-1]}. Available: {[a.next_state.facts[-1] for a in current_actions]}")
                        beam = [] # Empty beam to stop search
                        break

                else:
                    logging.error("No valid states left in beam to follow example solution.")
                    break # Stop if no valid state

                beam = next_beam_candidates # Update beam for next iteration (list of tuples)

            else: # Standard beam search expansion
                all_actions_in_step = []
                parent_map = {} # Map action object to parent state object

                # Iterate over the validated states and corresponding actions
                for parent_state, state_actions in zip(valid_beam_states, actions_per_state):
                    for action_idx, action in enumerate(state_actions):
                        assert isinstance(action, Action), \
                            f"BEAM EXP: Item {action_idx} in state_actions not Action: {type(action)}"
                        all_actions_in_step.append(action)
                        parent_map[action] = parent_state

                if not all_actions_in_step:
                    logging.info(f"No actions available from valid beam states at depth {i}. Stopping search.")
                    break

                # Add potential next states to global buffer
                if self.use_global_buffer:
                    next_states_to_buffer = [a.next_state for a in all_actions_in_step if a.next_state.facts[-1] not in seen]
                    self.add_to_global_buffer(next_states_to_buffer)

                # Query model
                try:
                    with torch.no_grad():
                        q_values = q(all_actions_in_step).tolist()
                except Exception as e:
                    logging.error(f"Error during Q-function call at depth {i}: {e}")
                    traceback.print_exc()
                    break

                # Prepare candidates for next beam
                candidates = [] # List to store (State, Action) tuples
                for action_idx, (action, value) in enumerate(zip(all_actions_in_step, q_values)):
                    assert isinstance(action, Action), \
                        f"CANDIDATE GEN: Item {action_idx} (action) not Action: {type(action)}"

                    action.value = value
                    next_state = action.next_state
                    parent_state = parent_map[action]
                    parent_val = parent_state.value if hasattr(parent_state, 'value') else 0.0
                    next_state.value = q.aggregate(parent_val, action.value)

                    if next_state.facts[-1] not in seen:
                        candidate_tuple = (next_state, action)
                        assert isinstance(candidate_tuple, tuple) and len(candidate_tuple)==2 and isinstance(candidate_tuple[0], State) and isinstance(candidate_tuple[1], Action), \
                            f"CANDIDATE GEN: Appending wrong format: {type(candidate_tuple)} {candidate_tuple}"
                        candidates.append(candidate_tuple)

                # Sort candidates
                candidates.sort(key=lambda item: item[0].value, reverse=True)

                # Build the next beam
                next_beam = [] # List to store (State, Action) tuples
                added_facts = set()

                # Epsilon-greedy or Greedy selection
                # Iterate over 'candidates', which we've asserted contains tuples
                if len(candidates) > self.beam_size and self.epsilon > 0:
                    num_rand = int(round(self.beam_size * self.epsilon))
                    num_top = self.beam_size - num_rand
                    count = 0

                    # Top candidates
                    for item_idx, item in enumerate(candidates):
                        assert isinstance(item, tuple) and len(item)==2, f"EPS TOP: Item {item_idx} not tuple: {item}"
                        state, action = item
                        if count >= num_top: break
                        if state.facts[-1] not in added_facts:
                            next_beam.append(item) # Append the tuple
                            added_facts.add(state.facts[-1])
                            seen.add(state.facts[-1])
                            current_step_visited.append(item)
                            count += 1
                            
                    # Random candidates
                    remaining_candidates = [(s, a) for s, a in candidates[count:] if s.facts[-1] not in added_facts]
                    num_rand_actual = min(num_rand, len(remaining_candidates))
                    if num_rand_actual > 0:
                        random_sample = random.sample(remaining_candidates, num_rand_actual)
                        for item_idx, item in enumerate(random_sample):
                            assert isinstance(item, tuple) and len(item)==2, f"EPS RAND: Item {item_idx} not tuple: {item}"
                            state, action = item
                            next_beam.append(item) # Append the tuple
                            added_facts.add(state.facts[-1])
                            seen.add(state.facts[-1])
                            current_step_visited.append(item)

                else: # Greedy
                    count = 0
                    for item_idx, item in enumerate(candidates):
                        assert isinstance(item, tuple) and len(item)==2, f"GREEDY: Item {item_idx} not tuple: {item}"
                        state, action = item
                        if count >= self.beam_size: break
                        if state.facts[-1] not in added_facts:
                            next_beam.append(item) # Append the tuple
                            added_facts.add(state.facts[-1])
                            seen.add(state.facts[-1])
                            current_step_visited.append(item)
                            count += 1

                beam = next_beam # Update beam for the next iteration

            # Add the visited states for this step to the overall list
            if current_step_visited:        # Avoid adding empty lists
                visited_states_with_actions.append(current_step_visited)

            # Logging the beam state
            for item_idx, item in enumerate(beam):
                assert isinstance(item, tuple) and len(item) == 2, \
                    f"BEAM CHECK (end loop): Item {item_idx} not (State, Action): {type(item)} {item}"
            beam_facts = [s.facts[-1] for s, _ in beam]      # Unpack after assertion
            logging.info(f'Beam #{i+1} (size {len(beam)}): {beam_facts}')

            if not beam:
                logging.info(f"Beam became empty after processing depth {i}. Stopping search.")
                break

        # Solution Found and Example Generation
        positive_path = [] # Stores Actions
        if solution is not None:
             # Add solution to stored solutions if not following an example
            if ex_solution is None and self.stored_solutions is not None:
                self.stored_solutions.append(solution) # Append the State object

            # Reconstruct path (using the parent_action links within the solution state)
            current_state_in_path = solution
            while current_state_in_path is not None and current_state_in_path.parent_action is not None:
                action_in_path = current_state_in_path.parent_action
                assert isinstance(action_in_path, Action), f"PATH RECONSTRUCTION: Expected Action, got {type(action_in_path)}"
                positive_path.append(action_in_path)
                current_state_in_path = action_in_path.state

            positive_path.reverse() # Order actions from start to goal

            # Create contrastive examples
            # Check consistency between path length and visited states list length
            if len(positive_path) + 1 != len(visited_states_with_actions):
                logging.warning(f"Path/Visited length mismatch ({len(positive_path)+1} vs {len(visited_states_with_actions)}). Skipping contrastive example gen.")
            else:
                for step_idx, positive_action in enumerate(positive_path):

                    # visited_states_with_actions[step_idx + 1] contains (state, action) tuples from the end of that step
                    visited_at_step_end = visited_states_with_actions[step_idx + 1]

                    beam_negatives = []
                    for visited_state, visited_action in visited_at_step_end:
                        assert visited_action is None or isinstance(visited_action, Action), \
                            f"CONTRASTIVE GEN: visited_action type error: {type(visited_action)}"
                        if visited_action and visited_action != positive_action:
                            if (visited_action.state == positive_action.state or
                                self.beam_negatives_frac >= random.random()):
                                beam_negatives.append(visited_action)

                    embedding_negatives = []
                    if self.use_embedding_based_sampling:
                         embedding_negatives = self.get_embedding_based_negatives(positive_action)

                    # Combine negatives (handle duplicates carefully if Actions aren't perfectly hashable by identity)
                    # For simplicity assuming Action objects are hashable for the set, or implement custom duplicate check
                    try:
                        all_negatives_set = {a for a in beam_negatives + embedding_negatives if a is not None}
                        all_negatives = list(all_negatives_set)
                    except TypeError: # Handle if Action not hashable
                        logging.warning("Action object not hashable for set deduplication in negatives. Using list with potential duplicates.")
                        all_negatives = [a for a in beam_negatives + embedding_negatives if a is not None]


                    if positive_action is not None:
                        example = ContrastiveExample(positive=positive_action,
                                                negatives=all_negatives,
                                                gap=1)
                        self.examples.append(example)
                    else:
                        logging.warning(f"Skipped contrastive example at step {step_idx} due to None positive_action.")


        logging.info('Solved? {} (solution len {}, q={})'
                     .format(solution is not None,
                             (len(positive_path) + 1) if solution else 'N/A',
                             type(q).__name__))

        return solution # Return the final State object representing the solution, or None

    def stats(self):
        base_stats = "{} solutions, {:.4f} training acc".format(
            self.training_problems_solved,
            self.training_acc_moving_average)

        # Add global buffer info if used
        if self.use_global_buffer:
            buffer_stats = ", global buffer size: {}".format(len(self.global_state_buffer))
            return base_stats + buffer_stats

        return base_stats

    def gradient_steps(self, env=None):
        if not self.examples:
            logging.warning("gradient_steps called but no examples in buffer.")
            return [] # Return empty list if no examples

        if not self.keep_optimizer:
            self.reset_optimizer()

        celoss = nn.CrossEntropyLoss()
        losses = []
        # Determine device from model parameters
        device = next(self.q_function.parameters()).device


        wrapper = tqdm.tqdm if self.optimize_every is None else lambda x: x
        for i in wrapper(range(self.n_gradient_steps)):
            # Check for evaluation trigger only if env and gd_evaluate_every are valid
            if env is not None and self.gd_evaluate_every is not None and self.gd_evaluate_every > 0 and i > 0 and i % self.gd_evaluate_every == 0:
                env.evaluate()

            # Sample example safely
            if not self.examples: break # Stop if buffer becomes empty during steps
            try:
                e = random.choice(self.examples)
            except IndexError:
                logging.warning("Could not sample from examples deque (likely empty). Stopping gradient steps.")
                break # Stop if sampling fails


            # Construct batch safely, handling potential None negatives
            valid_negatives = [neg for neg in e.negatives if neg is not None] # Filter out None actions

            # Limit number of negatives
            num_negatives_to_sample = min(self.max_negatives, len(valid_negatives))
            if num_negatives_to_sample < len(valid_negatives):
                sampled_negatives = random.sample(valid_negatives, num_negatives_to_sample)
            else:
                sampled_negatives = valid_negatives # Use all valid negatives if fewer than max_negatives

            # Ensure positive action is not None before proceeding
            if e.positive is None:
                logging.warning("Skipping gradient step due to None positive action in sampled example.")
                continue # Skip this gradient step

            all_actions = [e.positive] + sampled_negatives

            # Skip if batch is empty or only contains positive (shouldn't happen with valid example)
            if len(all_actions) <= 1 and not sampled_negatives:
                logging.warning(f"Skipping gradient step due to only positive action in batch (Example: {e.positive}).")
                continue

            try:
                self.optimizer.zero_grad()
                # Ensure q_function can handle the list of actions
                f_pred = self.q_function(all_actions)

                # Ensure f_pred is a valid tensor
                if not isinstance(f_pred, torch.Tensor) or f_pred.nelement() == 0:
                    logging.error(f"Q-function returned invalid output: {f_pred}")
                    continue # Skip step

                # Target is always class 0 (the positive example)
                target = torch.zeros(1, dtype=torch.long, device=device) # Use long dtype for CrossEntropyLoss

                # Calculate loss
                loss = celoss(f_pred.unsqueeze(0), target)

                # Check for NaN/inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"NaN or Inf loss detected: {loss.item()}. Skipping backward/step.")
                    # Optionally log details about f_pred or actions
                    continue # Skip backprop and step

                wandb.log({'train_loss': loss.item()}) 
                losses.append(loss.item())
                loss.backward()

                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.q_function.parameters(), max_norm=1.0)

                self.optimizer.step()

            except Exception as err:
                logging.error(f"Error during gradient step {i}: {err}")
                traceback.print_exc() # Print full traceback
                # Consider whether to continue or break loop on error
                # For robustness, let's continue to the next step for now
                continue


        del e, all_actions, f_pred, loss, target, valid_negatives, sampled_negatives
        if 'q_values' in locals(): del q_values

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log average loss if any steps were successful
        if losses:
            avg_loss = sum(losses) / len(losses)
            logging.info(f"Finished {len(losses)} gradient steps with avg loss: {avg_loss:.4f}")
            wandb.log({'avg_train_loss_batch': avg_loss}) # Log average loss for the batch of steps


        return losses

@register(LearningAgent)
class BeamSearchIterativeDeepening(LearningAgent):
    def __init__(self, q_function, config):
        self.q_function = q_function
        self.bootstrapping = True
        self.replay_buffer_size = config['replay_buffer_size']

        self.replay_buffer_pos = collections.deque(maxlen=self.replay_buffer_size)
        self.replay_buffer_neg = collections.deque(maxlen=self.replay_buffer_size)
        self.training_problems_solved = 0

        self.max_depth = config['max_depth']
        self.depth_step = config['depth_step']
        self.initial_depth = config['initial_depth']
        self.step_every = config['step_every']
        self.beam_size = config['beam_size']
        self.beam_negatives = config.get('beam_negatives', True)

        self.balance_examples = config.get('balance_examples', True)
        self.optimize_on = config.get('optimize_on', 'problem')
        self.reward_decay = config.get('reward_decay', 1.0)
        self.batch_size = config.get('batch_size', 64)
        self.optimize_every = config.get('optimize_every', 1)
        self.n_gradient_steps = config.get('n_gradient_steps', 10)
        self.discard_unsolved_problems = config.get('discard_unsolved', False)
        self.full_imitation_learning = config.get('full_imitation_learning', False)

        if config.get('bootstrap_from', 'Random') == 'InverseLength':
            self.bootstrap_policy = InverseLength(self.q_function.device)
        else:
            self.bootstrap_policy = RandomQFunction(self.q_function.device)

        # Knob: whether to add an artificial 'success' state in the end
        # of the solution in training examples. The idea is that this would align
        # all states that are in the path to a solution closer together.
        self.add_success_state = config.get('add_success_state', False)
        # Knob: how many future states to use as examples.
        self.n_future_states = config.get('n_future_states', 1)
        self.n_negatives = config.get('n_negatives', 1)
        self.learning_rate = config.get('lr', 1e-4)

        self.current_depth = self.initial_depth
        self.bootstrapping = True

    def name(self):
        if self.full_imitation_learning:
            return 'ImitationLearning'
        elif self.depth_step == 0 and not self.balance_examples:
            return 'DAgger'
        elif self.depth_step > 0 and not self.balance_examples:
            return 'IDDagger'
        elif self.depth_step > 0 and self.balance_examples:
            return 'IDCDagger'

    def learn_from_environment(self, environment):
        for i in itertools.count():
            problem = environment.generate_new()
            solution = self.beam_search(problem, environment)

            if solution is not None:
                self.training_problems_solved += 1

            if ((self.optimize_on == 'problem' and (i + 1) % self.optimize_every == 0) or
                (self.optimize_on == 'solution' and solution is not None and
                 self.training_problems_solved % self.optimize_every == 0)):
                logging.info('Running SGD steps.')
                self.gradient_steps()

            if (i + 1) % self.step_every == 0:
                self.current_depth = min(self.max_depth, self.current_depth + self.depth_step)
                logging.info(f'Beam search depth increased to {self.current_depth}.')

    # def learn_from_environment(self, environment):
    #     ex_sol_left = True if self.example_solutions and self.training_problems_explored < len(self.example_solutions) else False
    #     if self.bootstrapping and self.training_problems_solved >= self.n_bootstrap_problems:
    #         print("Bootstrap phase complete.")
    #         self.bootstrapping = False

    #     # REMOVE the outer try block around the loop
    #     # try: # <-- REMOVE
    #     for i in itertools.count():
    #         # Check if using example solutions
    #         if ex_sol_left:
    #             if self.training_problems_explored >= len(self.example_solutions):
    #                 print("Finished all example solutions.")
    #                 ex_sol_left = False
    #                 # If max_steps is the only limit, we need to stop if examples run out
    #                 if environment.max_steps is None:
    #                      print("Finished examples and no max_steps defined. Stopping learn_from_environment.")
    #                      break # Exit the loop if no more examples and no step limit
    #                 s = environment.generate_new()
    #                 ex_solution = None
    #             else:
    #                 ex_solution = self.example_solutions[self.training_problems_explored]
    #                 s = environment.generate_new(seed=ex_solution.problem_seed) # Use seed from example
    #                 print(f"Using example solution {self.training_problems_explored}: {s.facts[-1]}")
    #         else:
    #             s = environment.generate_new()
    #             ex_solution = None

    #         self.training_problems_explored += 1

    #         solution = self.beam_search(s, environment, ex_solution=ex_solution)

    #         if solution is not None:
    #             self.training_problems_solved += 1
    #             # Update moving average accuracy
    #             self.training_acc_moving_average = self.training_acc_moving_average * 0.99 + 0.01

    #             # Add positive example if using contrastive loss
    #             if hasattr(self, 'add_positive_example'):
    #                 self.add_positive_example(solution)

    #         else: # Failed to find solution
    #             # Update moving average accuracy
    #             self.training_acc_moving_average = self.training_acc_moving_average * 0.99

    #         # Check if bootstrapping is complete
    #         if self.bootstrapping and self.training_problems_solved >= self.n_bootstrap_problems:
    #             print("Bootstrap phase complete.")
    #             self.bootstrapping = False

    #         # Check if optimization step is needed
    #         if self.optimize_every is not None and self.training_problems_explored % self.optimize_every == 0:
    #             if not self.bootstrapping: # Only optimize after bootstrapping
    #                 losses = self.gradient_steps(environment)
    #                 if losses:
                        # Log average loss if needed (e.g., wandb.log({'train_loss': np.mean(losses)}))
    #                     pass

    #         # Check if depth needs stepping (only relevant if not following examples strictly)
    #         if self.step_every is not None and self.step_every > 0 and (self.training_problems_explored + 1) % self.step_every == 0:
    #              if self.depth_step is not None and self.depth_step > 0: # Check if depth_step is defined and positive
    #                  self.current_depth = min(self.max_depth, self.current_depth + self.depth_step)
    #                  logging.info(f'Beam search depth increased to {self.current_depth}.')
    #                  print(f'Beam search depth increased to {self.current_depth}.')
    #     print("Exiting NCE.learn_from_environment.") # This might not be reached if EndOfLearning is raised

    def learn_from_experience(self):
        if self.full_imitation_learning:
            logging.info('Running Imitation learning')
            self.gradient_steps(True)

    def get_q_function(self):
        if self.bootstrapping:
            return self.bootstrap_policy
        return self.q_function

    def beam_search(self, state, environment):
        '''Performs beam search in a train problem while recording particular examples
        in the replay buffer (according to the various knobs in the algorithm, see config)'''

        states_by_id = {id(state): state}
        state_parent_edge = {}
        beam = [state]
        solution = None  # The state that we found that solves the problem.
        action_reward = {}  # Remember rewards we attribute to each action.
        q = self.get_q_function()
        seen = {state}

        logging.info(f'Trying {state}')

        for i in range(self.current_depth):
            rewards, actions = zip(*environment.step(beam))

            for s, r, state_actions in zip(beam, rewards, actions):
                for a in state_actions:
                    # Remember how we got to this state.
                    states_by_id[id(a.next_state)] = a.next_state
                    state_parent_edge[id(a.next_state)] = (s, a)
                # Record solution, if found.
                if r:
                    if self.add_success_state:
                        success = copy.deepcopy(SUCCESS_STATE)
                        a = Action(s, 'success', success, 1.0, 1.0)
                        success.parent_action = a
                        states_by_id[id(success)] = success
                        state_parent_edge[id(success)] = (s, a)
                        solution = [success]
                    else:
                        solution = [s]

            if solution is not None:
                # Traverse all the state -> next_state edges backwards, remembering
                # all states in the path to the solution.
                current = solution[0]
                current_reward = 1.0

                while id(current) in state_parent_edge:
                    prev_s, a = state_parent_edge[id(current)]
                    action_reward[id(a)] = current_reward
                    current_reward *= self.reward_decay
                    current = prev_s
                    solution.append(current)

                solution = list(reversed(solution))
                break

            all_actions = [a for state_actions in actions for a in state_actions]

            if not len(all_actions):
                break

            # Query model, sort next states by value, then update beam.
            with torch.no_grad():
                q_values = q(all_actions).tolist()

            for a, v in zip(all_actions, q_values):
                a.value = v

            next_states = []
            for s, state_actions in zip(beam, actions):
                for a in state_actions:
                    ns = a.next_state
                    ns.value = s.value + np.log(a.value)
                    next_states.append(ns)

            next_states.sort(key=lambda s: s.value, reverse=True)
            # Remove duplicates while keeping the order (i.e. if a state appears multiple times,
            # keep the one with the largest value). Works because dict is ordered in Python 3.6+.
            next_states = [s for s in dict.fromkeys(next_states) if s not in seen]
            seen.update(next_states)
            beam = next_states[:self.beam_size]
            logging.info(f'Beam #{i}: {beam}:')

            if not beam:
                break

        logging.info('Solved? {} (solution len {}, q={})'
                     .format(solution is not None,
                             solution and len(solution),
                             type(q)))

        # Add all edges traversed as examples in the experience replay buffer.
        if solution is not None or not self.discard_unsolved_problems:
            positive_ids = set(id(s) for s in solution) if solution is not None else set()
            # Add negative examples.
            for s, (parent, a) in state_parent_edge.items():
                r = action_reward.get(id(a), 0.0)
                if r == 0 and (self.beam_negatives or id(s) in positive_ids):
                    self.replay_buffer_neg.append((states_by_id[s], a, 0))
            # Add positive examples (possibly looking several steps ahead, depending
            # on `self.n_future_states`.
            if solution is not None:
                for i, s_i in enumerate(solution):
                    for j in range(i+1, min(i + 1 + self.n_future_states, len(solution))):
                        s_j = solution[j]
                        self.replay_buffer_pos.append((states_by_id[s],
                                                       s_j.parent_action,
                                                       action_reward[id(s_j.parent_action)]))

        return None if solution is None else solution[-1]

    def stats(self):
        return "replay buffer size = {}, {} positive".format(
            len(self.replay_buffer_pos) + len(self.replay_buffer_neg),
            len(self.replay_buffer_pos))

    def gradient_steps(self, is_last_round=False):
        if self.full_imitation_learning and not is_last_round:
            return

        if self.balance_examples:
            n_pos = len(self.replay_buffer_pos)
            n_neg = min(self.n_negatives * n_pos, len(self.replay_buffer_neg))
            examples = (random.sample(self.replay_buffer_pos, k=n_pos) +
                        random.sample(self.replay_buffer_neg, k=n_neg))
        else:
            examples = self.replay_buffer_pos + self.replay_buffer_neg

        logging.info(f'Taking {self.n_gradient_steps} with {len(examples)} examples'
                     f' (balanced = {self.balance_examples})')
        batch_size = min(self.batch_size, len(examples))

        if batch_size == 0:
            return

        optimizer = torch.optim.Adam(self.q_function.parameters(), lr=self.learning_rate)

        for i in range(self.n_gradient_steps):
            batch = random.sample(examples, batch_size)
            batch_s, batch_a, batch_r = zip(*batch)

            optimizer.zero_grad()

            r_pred = self.q_function(batch_a)
            loss = F.binary_cross_entropy(r_pred, torch.tensor(batch_r,
                                                               dtype=r_pred.dtype,
                                                               device=r_pred.device))
            wandb.log({'train_loss': loss.item()})
            loss.backward()
            optimizer.step()

        self.bootstrapping = False

        gc.collect()
        torch.cuda.empty_cache()


# A tuple of the replay buffer. We don't need to store the current state or the next state
# because a0 is an Action object, which already has a0.state and a0.next_state.
QReplayBufferTuple = collections.namedtuple('QReplayBufferTuple',
                                            ['a0', 'r', 'A1'])


@register(LearningAgent)
class QLearning(LearningAgent):
    def __init__(self, q_function, config):
        self.q_function = q_function

        self.replay_buffer_size = config['replay_buffer_size']
        self.max_depth = config['max_depth']

        self.discount_factor = config.get('discount_factor', 1.0)
        self.batch_size = config.get('batch_size', 64)
        self.optimize_every = config.get('optimize_every', 16)
        self.softmax_alpha = config.get('softmax_alpha', 1.0)

        self.replay_buffer = collections.deque(maxlen=self.replay_buffer_size)
        self.solutions_found = 0

        self.optimizer = torch.optim.Adam(q_function.parameters(),
                                          lr=config.get('learning_rate', 1e-4))

    def name(self):
        return 'QLearning'

    def get_q_function(self):
        return self.q_function

    def learn_from_environment(self, environment):
        for i in itertools.count():
            state = environment.generate_new()
            r, actions = environment.step([state])[0]

            if r:
                # Trivial state: already solved, no examples to draw.
                continue

            for j in range(self.max_depth):
                # No actions to take.
                if not len(actions):
                    break

                with torch.no_grad():
                    q_values = self.q_function(actions)
                    pi = Categorical(logits=self.softmax_alpha * q_values)
                    a = pi.sample().item()

                s_next = actions[a].next_state
                r, next_actions = environment.step([s_next])[0]
                self.replay_buffer.append(QReplayBufferTuple(actions[a],
                                                             r,
                                                             next_actions))

            if i % self.optimize_every == 0:
                self.gradient_steps()

    def learn_from_experience(self):
        pass  # QLearning doesn't have a learning step at the end.

    def stats(self):
        return "replay buffer size = {}, {} solutions found".format(
            len(self.replay_buffer), self.solutions_found)

    def gradient_steps(self):
        examples = self.replay_buffer
        batch_size = min(self.batch_size, len(examples))

        if batch_size == 0:
            return

        batch = random.sample(examples, batch_size)
        ys = []

        # Compute ys.
        with torch.no_grad():
            for t in batch:
                if t.r > 0 or not t.A1:  # Next state is terminal.
                    ys.append(t.r)
                else:
                    # Need to compute maximum Q value for all actions.
                    max_q = self.q_function(t.A1).max()
                    ys.append(t.r + self.discount_factor * max_q)

        # Compute Q estimates and take gradient steps.
        self.optimizer.zero_grad()
        q_estimates = self.q_function([t.a0 for t in batch])

        y = torch.tensor(ys, dtype=q_estimates.dtype, device=q_estimates.device)
        loss = ((y - q_estimates)**2).mean()
        wandb.log({'train_loss': loss.item()})
        loss.backward()
        self.optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()


@register(LearningAgent)
class AutodidaticIteration(LearningAgent):
    def __init__(self, q_function, config):
        self.q_function = q_function

        self.batch_size = config.get('batch_size', 64)
        self.n_gradient_steps = config.get('gradient_steps', 64)
        self.reward_weight = config.get('reward_weight', 0.02)

        self.optimizer = torch.optim.Adam(q_function.parameters(),
                                          lr=config.get('learning_rate', 1e-4))
        self.examples = []

    def name(self):
        return 'AutodidaticIteration'

    def get_q_function(self):
        return self.q_function

    def learn_from_environment(self, environment):
        for i in itertools.count():
            states = [environment.generate_new() for _ in range(self.batch_size)]

            for s in states:
                r, actions = environment.step([s])[0]

                if r:
                    # Trivial state; no examples to draw.
                    continue

                with torch.no_grad():
                    q_s = self.q_function(actions)

                r_a = []

                for i, a in enumerate(actions):
                    r_a_i, _ = environment.step([a.next_state])[0]
                    r_a.append(float(r_a_i) + q_s[i].item())

                value = np.max(r_a) - self.reward_weight
                self.examples.append((s, value))

            self.gradient_steps()

    def learn_from_experience(self):
        pass

    def stats(self):
        return f"n_training_examples={len(self.examples)}"

    def gradient_steps(self):
        examples = self.examples
        batch_size = min(self.batch_size, len(examples))

        if batch_size == 0:
            return

        batch = random.sample(examples, batch_size)

        for _ in range(self.n_gradient_steps):
            y_p = self.q_function([Action(st, '', st, 0.0, 0.0) for st, _ in batch])
            y = torch.tensor([y for _, y in batch], dtype=y_p.dtype, device=y_p.device)

            self.optimizer.zero_grad()
            loss = ((y_p - y)**2).mean()
            loss.backward()
            wandb.log({'train_loss': loss.item()})
            self.optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()


@register(LearningAgent)
class DAVI(LearningAgent):
    def __init__(self, q_function, config):
        self.q_function = q_function

        self.batch_size = config.get('batch_size', 64)
        self.n_gradient_steps = config.get('gradient_steps', 64)
        self.g = config.get('step_cost', 0.01)

        self.optimizer = torch.optim.Adam(q_function.parameters(),
                                          lr=config.get('learning_rate', 1e-4))
        self.examples = []

    def name(self):
        return 'DAVI'

    def get_q_function(self):
        return self.q_function

    def learn_from_environment(self, environment):
        for i in itertools.count():
            states = [environment.generate_new() for _ in range(self.batch_size)]

            for s in states:
                r, actions = environment.step([s])[0]

                if r:
                    self.examples.append((s, 0))
                    continue

                with torch.no_grad():
                    q_s = self.q_function(actions)

                r_a = []

                for i, a in enumerate(actions):
                    r_a_i, _ = environment.step([a.next_state])[0]
                    r_a.append(self.g + (0 if r_a_i else q_s[i].item()))

                value = np.min(r_a)
                self.examples.append((s, value))

            self.gradient_steps()

    def learn_from_experience(self):
        pass

    def stats(self):
        return f"n_training_examples={len(self.examples)}"

    def gradient_steps(self):
        examples = self.examples
        batch_size = min(self.batch_size, len(examples))

        if batch_size == 0:
            return

        batch = random.sample(examples, batch_size)

        for _ in range(self.n_gradient_steps):
            y_p = self.q_function([Action(st, '', st, 0.0, 0.0) for st, _ in batch])
            y = torch.tensor([y for _, y in batch], dtype=y_p.dtype, device=y_p.device)

            self.optimizer.zero_grad()
            loss = ((y_p - y)**2).mean()
            loss.backward()
            wandb.log({'train_loss': loss.item()})
            self.optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()

@register(LearningAgent)
class BehavioralCloning(LearningAgent):
    def __init__(self, q_function, config):
        self.q_function = q_function

        self.batch_size = config.get('batch_size', 64)
        self.n_gradient_steps = config.get('gradient_steps', 5000)
        self.max_depth = config.get('max_depth', 20)

        self.optimizer = torch.optim.Adam(q_function.parameters(),
                                          lr=config.get('learning_rate', 1e-4))
        self.examples = []

    def name(self):
        return 'BehavioralCloning'

    def get_q_function(self):
        return self.q_function

    def learn_from_environment(self, environment):
        for i in itertools.count():
            s = environment.generate_new()
            r, actions = environment.step([s])[0]

            available_actions = []
            solution = []

            if r:
                # Trivial state; no examples to draw.
                continue

            for i in range(self.max_depth):
                if r or len(actions) == 0:
                    break
                available_actions.append(actions)
                a = random.randint(0, len(actions) - 1)
                solution.append(actions[a])
                r, actions = environment.step([solution[-1].next_state])[0]

            if r:
                for actions, answer in zip(available_actions, solution):
                    self.examples.append((actions, answer))

    def stats(self):
        return f"n_training_examples={len(self.examples)}"

    def learn_from_experience(self):
        examples = self.examples
        batch_size = min(self.batch_size, len(examples))

        if batch_size == 0:
            return

        celoss = nn.CrossEntropyLoss()
        losses = []

        for i in range(self.n_gradient_steps):
            (all_actions, answer) = random.choice(self.examples)
            self.optimizer.zero_grad()
            f_pred = self.q_function(all_actions)
            loss = celoss(f_pred.unsqueeze(0), answer * torch.ones(1, dtype=int, device=f_pred.device))
            wandb.log({'train_loss': loss.item()})
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()


def run_agent_experiment(config, device, resume):
    experiment_id = config['experiment_id']
    domain = config['domain']
    agent_name = config['agent']['name']
    run_index = config.get('run_index', 0)

    run_id = "{}-{}-{}{}".format(experiment_id, agent_name, domain, run_index)

    wandb.init(id=run_id,
               name=run_id,
               config=config,
               entity='ritishtest1',
               project=config.get('wandb_project', 'test'),
               reinit=True,
               resume=resume)

    env = Environment.from_config(config)
    q_fn = QFunction.new(config['agent']['q_function'], device)
    agent = LearningAgent.new(q_fn, config['agent'])

    eval_env = EnvironmentWithEvaluationProxy(experiment_id, run_index, agent_name, domain,
                                              agent, env, config['eval_environment'])
    eval_env.evaluate_agent()


def learn_abstract(config, device, resume):
    experiment_id = config['experiment_id']
    domain = config['domain']
    agent_name = config['agent']['name']
    run_index = config.get('run_index', 0)
    assert 'compression' in config and 'iterations' in config
    AbsType = ABS_TYPES[config['compression']['abs_type']]

    run_id = "{}-{}-{}{}".format(experiment_id, agent_name, domain, run_index)

    wandb.init(id=run_id,
               name=run_id,
               config=config,
               entity='ritishtest1',
               project=config.get('wandb_project', 'test'),
               reinit=True,
               resume=resume)

    restart_count = False
    subrun_index = 0
    while True:
        # LEARNING
        try:
            env = Environment.from_config(config)
            q_fn = QFunction.new(config['agent']['q_function'], device)
            assert 'num_store_sol' in config['agent']
            agent = LearningAgent.new(q_fn, config['agent'])
            # config['eval_environment']['eval_config']['seed'] = random.randint(200_000_000, 300_000_000)
            config['eval_environment']['restart_count'] = restart_count
            eval_env = EnvironmentWithEvaluationProxy(experiment_id, run_index, agent_name, domain,
                                                      agent, env, config['eval_environment'], subrun_index)
            print("MAX NEGATIVES:", eval_env.agent.max_negatives)
            subrun_index = eval_env.subrun_index
            if 'max_steps_list' in config['eval_environment']:
                eval_env.max_steps = config['eval_environment']['max_steps_list'][subrun_index]
            if 'success_thres_list' in config['eval_environment']:
                eval_env.success_thres = config['eval_environment']['success_thres_list'][subrun_index]
        except RuntimeError:
            # manually reconstruct ckpt if it's broken (e.g. b/c device ran out of memory)
            print("CHECKPOINT BROKEN... MANUAL RECONSTRUCTION OF CHECKPOINT")
            temp_config = {
                    'environment_backend': 'Rust',
                    'abstractions': {
                        'abs_ax': [("refl"), ("comm"), ("assoc"), ("dist"), ("sub_comm"), ("eval"), ("add0"), ("sub0"), ("mul1"), ("div1"), ("div_self"), ("sub_self"), ("subsub"), ("mul0"), ("zero_div"), ("add"), ("sub"), ("mul"), ("div"), ("assoc~eval:_1"), ("eval~mul1:1_"), ("eval~eval:0_"), ("div~assoc:$_0.0"), ("comm~assoc:0_"), ("eval~mul1:1_0"), ("eval~assoc:1_0"), ("eval~eval:0.1_1")]
                    },
                    'domain': 'equations-ct'
            }
            q_fn_path = "output/loop_1m/ConPoLe/equations-ct/run0/checkpoints/1-21.pt"
            device = torch.device(0)
            env = Environment.from_config(temp_config)
            q_fn = torch.load(q_fn_path, map_location=device)
            q_fn.to(device)
            agent_config = {
                "type": "NCE",
                "name": "ConPoLe",
                "n_future_states": 1,
                "replay_buffer_size": 100000,
                "max_depth": 30,
                "beam_size": 10,
                "initial_depth": 30,
                "depth_step": 0,
                "optimize_every": 16,
                "n_gradient_steps": 128,
                "keep_optimizer": True,
                "step_every": 10000,
	        "epsilon": 0.2,
                "q_function": {
                    "type": "Bilinear",
                    "char_emb_dim": 64,
                    "hidden_dim": 256,
                    "mlp": True,
                    "lstm_layers": 2
                }
            }
            agent = LearningAgent.new(q_fn, agent_config)
            agent.bootstrapping = False
            assert 'num_store_sol' in config['eval_environment']
            config['eval_environment']['eval_config']['seed'] = random.randint(200_000_000, 300_000_000)
            config['eval_environment']['restart_count'] = restart_count
            eval_env = EnvironmentWithEvaluationProxy(experiment_id, run_index, agent_name, domain,
                                                      agent, env, config['eval_environment'], subrun_index, try_load_ckpt=False)
            eval_env.n_steps = 2200003
            eval_env.n_new_problems = 23317
            eval_env.cumulative_reward = 8137
            eval_env.n_checkpoints = 22
            eval_env.subrun_index = 1
            subrun_index = eval_env.subrun_index
        begin_time = datetime.now()
        print(f"ITERATION {subrun_index} TRAINING BEGINS AT {begin_time}")
        print("AXIOMS AND ABSTRACTIONS:", eval_env.environment.rules)
        print(f"USING {0 if eval_env.agent.example_solutions is None else len(eval_env.agent.example_solutions)} EXAMPLE SOLUTIONS")
        eval_env.evaluate_agent()
        end_time = datetime.now()
        print(f"ITERATION {subrun_index} TRAINING COMPLETE AT {end_time}; TOTAL TRAINING TIME {end_time-begin_time}")
        if subrun_index >= config['iterations'] - 1:
            break

        # ABSTRACTING
        raw_solutions = eval_env.agent.stored_solutions
        solutions = []
        # Convert to format for abstraction algo
        for raw_sol in raw_solutions:
            states = raw_sol.facts
            actions = []
            action = raw_sol.parent_action
            while action is not None:
                actions.append(steps.Step.from_string(action.action, AbsType, action.next_state.facts[-2:]))
                action = action.state.parent_action
            solutions.append(steps.Solution(states, list(reversed(actions))))
        
        begin_time = datetime.now()
        print(f"ITERATION {subrun_index} ABSTRACTING BEGINS AT {begin_time}")
        print(f"USING {len(solutions)} SOLUTIONS")
        abs_ax = eval_env.environment.rules
        if abs_ax is None:
            abs_ax = [Axiom(ax_str, AbsType) for ax_str in AXIOMS[domain]]
        compressor = COMPRESSORS[config['compression']['compressor']](solutions, abs_ax, config['compression'])
        num_iter, num_abs_sol = config['compression'].get('iter', 1), config['compression'].get('num_abs_sol')
        abs_sols, abs_ax = compressor.iter_abstract(num_iter, True, num_abs_sol)
        end_time = datetime.now()
        print(f"ITERATION {subrun_index} ABSTRACTING COMPLETE AT {end_time}; TOTAL ABSTRACTING TIME {end_time-begin_time}")

        config['abstractions'] = {'abs_ax': abs_ax, 'abs_type': config['compression']['abs_type']}
        config['agent']['example_solutions'] = abs_sols
        abs_ax_path = os.path.join(eval_env.checkpoint_dir, f'A{subrun_index}.pkl')
        with open(abs_ax_path, 'wb') as f:
            pickle.dump(abs_ax, f)
        abs_sols_path = os.path.join(eval_env.checkpoint_dir, f'AS{subrun_index}.pkl')
        with open(abs_sols_path, 'wb') as f:
            pickle.dump(abs_sols, f)

        restart_count = True
        subrun_index += 1


def run_batch_experiment(config, range_to_run):
    'Spawns a series of processes to run experiments for each agent/domain pair.'
    experiment_id = config.get('experiment_id', util.random_id())
    domains = config['domains']
    agents = [c for c in config['agents'] if not c.get('disable')]
    n_runs = config.get('n_runs', 1)

    environment_backend = config.get('environment_backend', 'Racket')
    environment_port_base = config.get('environment_port_base', 9876)
    port = 0

    environment_url = config.get('environment_url')
    _, port = environment_url.split(':') if environment_url else (None, None)
    if port is not None:
        environment_port_base = int(port)

    run_processes = []
    environments = []
    agent_index = 0
    gpus = config.get('gpus', [])

    if not gpus:
        print('WARNING: no GPUs specified.')

    print('Starting experiment', experiment_id)

    try:
        for domain in domains:
            for agent in agents:
                print(f'Running {agent["name"]} on {domain}')

                for run_index in range(n_runs):
                    if agent_index < range_to_run[0] or agent_index >= range_to_run[1]:
                        print(f'Run {agent_index} not in range - skipping')
                        agent_index += 1
                        continue

                    if environment_backend == 'Racket':
                        port = environment_port_base + agent_index
                        environment_process = subprocess.Popen(
                            ['racket', 'environment.rkt', '-p', str(port)],
                            stderr=subprocess.DEVNULL)
                        environments.append(environment_process)

                        # Wait for environment to be ready.
                        time.sleep(0)

                    run_config = {
                        'experiment_id': experiment_id,
                        'run_index': run_index,
                        'agent': agent,
                        'domain': domain,
                        'environment_backend': environment_backend,
                        'environment_url': 'http://localhost:{}'.format(port),
                        'multitask_train_domains': config.get('multitask_train_domains'),
                        'eval_environment': copy.deepcopy(config['eval_environment']),
                        # # 'wandb_project': config.get('wandb_project')
                    }

                    has_abstractions = agent.get('compression') is not None

                    if has_abstractions:
                        run_config['compression'] = agent['compression']
                        run_config['iterations'] = agent['iterations']
                        command = 'learn-abstract'
                    else:
                        command = 'learn'

                    print('Running agent with config', json.dumps(run_config))

                    agent_process = subprocess.Popen(
                        ['python3', 'agent_rust.py', f'--{command}', '--config', json.dumps(run_config)]
                        + (['--gpu', str(gpus[agent_index % len(gpus)])] if gpus else []),
                        stderr=subprocess.DEVNULL)
                    run_processes.append(agent_process)

                    agent_index += 1

        print('Waiting for all agents to finish...')
        for p in run_processes:
            p.wait()
        print('Shutting down environments...')
        for p in environments:
            p.terminate()
        print('Done!')

    except (Exception, KeyboardInterrupt):
        print('Killing all created processes...')
        for p in run_processes + environments:
            p.terminate()

        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train RL agents to solve symbolic domains")
    parser.add_argument('--config', help='Path to config file, or inline JSON.', required=True)
    parser.add_argument('--learn', help='Put an agent to learn from the environment', action='store_true')
    parser.add_argument('--learn-abstract', help='Automatically loop between learning and abstracting', action='store_true')
    parser.add_argument('--experiment', help='Run a batch of experiments with multiple agents and environments',
                        action='store_true')
    parser.add_argument('--eval', help='Evaluate a learned policy', action='store_true')
    parser.add_argument('--eval-checkpoints', help='Show the evolution of a learned policy during interaction',
                        action='store_true')
    parser.add_argument('--debug', help='Enable debug messages.', action='store_true')
    parser.add_argument('--range', type=str, default=None,
                        help='Range of experiments to run. Format: 2-5 means range [2, 5).'
                        'Used to split experiments across multiple machines. Default: all')
    parser.add_argument('--gpu', type=int, default=None, help='Which GPU to use.')
    parser.add_argument('--resume', help='Resume a run that crashed.', action='store_true')

    opt = parser.parse_args()

    try:
        if opt.config:
            config = json.loads(opt.config)
    except json.decoder.JSONDecodeError:
        config = json.load(open(opt.config))

    device = torch.device('cpu') if opt.gpu is None else torch.device(opt.gpu)
    torch.cuda.empty_cache()

    # configure logging.
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)

    if opt.debug:
        logging.getLogger().setLevel(logging.INFO)

    if opt.range:
        range_to_run = tuple(map(int, opt.range.split('-')))
    else:
        range_to_run = (0, 10**9)

    # Only shown in debug mode.
    logging.info('Running in debug mode.')

    opt.resume = config.get('resume', "allow")

    if opt.learn:
        run_agent_experiment(config, device, opt.resume)
    elif opt.learn_abstract:
        learn_abstract(config, device, opt.resume)
        # import cProfile
        # cProfile.run('learn_abstract(config, device, opt.resume)', 'prostats')
    elif opt.eval:
        evaluate_policy(config, device, config.get('verbose', False))
    elif opt.eval_checkpoints:
        evaluate_policy_checkpoints(config, device)
    elif opt.experiment:
        run_batch_experiment(config, range_to_run)

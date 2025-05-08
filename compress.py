from __future__ import annotations
"""
Simple compression algorithm that finds common subsequences of actions and abstracts them.
"""

import sys
import json
import pickle
import argparse
import warnings
import math
import gym
import numpy as np
from gym import spaces
import random
import gym
import torch
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import json
from collections import defaultdict
from dataclasses import dataclass
import heapq
import wandb
from wandb.integration.sb3 import WandbCallback

import itertools

from datetime import datetime
import doctest

from steps import Step, AbsStep, Solution
from abstractions import Axiom, Abstraction, ABS_TYPES, AxSeqTreeRelPos,AxiomSeq
import abs_util


def log(x):
    ' For entropy calculations '
    return -1000. if x == 0 else math.log(x)


class Compress:
    def __init__(self, solutions, axioms, config):
        self.solutions = solutions

        self.AbsType = ABS_TYPES[config["abs_type"]]
        self.num_ax = len(axioms)  # num of axioms
        self.axioms = axioms  # list of Rule objects (Axiom objects during first cycle)
        self.axiom_index = {self.axioms[k]: k for k in range(self.num_ax)}  # dictionary mapping axiom names to their indices (as in the list self.axioms)
        self.abstractions = None
        self.new_axioms = self.axioms.copy()  # list containing axioms + additional actions as abstractions (called "new axioms")
        self.new_axiom_set = set(self.new_axioms)
        self.abstracted_solutions = None

        self.frequencies = None
        self.config = config
        self.peek_pos = config.get("peek_pos", False)

        self.max_abs_len = None

    def abstract(self):
        """
        Returns list of Abstraction objects
        """
        raise NotImplementedError

    def abstract_step(self, solution, abs_len, abstractions):
        """
        In solution, abstract out the first length-'abs_len' subsequence that is an abstraction
        solution: Solution object
        """
        i = 0
        while i <= len(solution.actions) - abs_len:
            ax_subseq = solution.actions[i:i+abs_len]
            new_ax = self.AbsType.from_steps(ax_subseq)
            if new_ax in abstractions:
                new_states = solution.states[:i+1] + solution.states[i+abs_len:]
                new_actions = solution.actions[:i] + [AbsStep(ax_subseq, self.AbsType, solution.states[i:i+abs_len+1])] + solution.actions[i+abs_len:]
                solution = Solution(new_states, new_actions)
            i += 1
        return solution
    
    def abstracted_sol(self, num_new_sols=None):
        """
        Get abstracted solutions from set of abstractions
        Format: same as self.solutions
        (i.e. tuple of Solution objects)
        """
        abstractions = self.abstractions or self.abstract()
        max_len = max(len(abstraction.rules) for abstraction in abstractions) if self.max_abs_len is None else self.max_abs_len
        
        self.new_axioms += abstractions
        abs_set = set(abstractions)
        self.new_axiom_set |= abs_set

        if (sols := self.abstracted_solutions) is not None:
            return sols if num_new_sols is None else sols[:num_new_sols]

        new_sols = self.solutions.copy() if num_new_sols is None else self.solutions[:num_new_sols]
        for abs_len in range(max_len, 1, -1):
            for i in range(len(new_sols)):
                new_sols[i] = self.abstract_step(new_sols[i], abs_len, abs_set)
        return new_sols


class IterAbsPairs(Compress):
    def __init__(self, solutions, axioms, config):
        super().__init__(solutions, axioms, config)
        self.thres = config.get("thres")
        self.top = config.get("top")
        self.only_flat = config.get("only_flat", False)
        self.max_abs_len = 2

    def get_frequencies(self, factor=1, max_len=2, min_len=2, mask=None):
        """
        Gets frequencies of (current action, next action) pairs
        Also stores example abstraction for each rel. pos. if --peek
        `factor` is factor to divide no. of occurrences by
        Disallow abstractions containing rules in `mask` (iterable)
        """
        if mask is None and not self.only_flat:
            mask_list = [[(0, len(sol.actions))] for sol in self.solutions]
        else:
            if mask is None:
                mask = set()
            mask_list = [[] for _ in self.solutions]
            for i, sol in enumerate(self.solutions):
                in_legal_zone = False
                for j, action in enumerate(sol.actions):
                    if action.rule in mask or (self.only_flat and len(action) != 1):
                        if in_legal_zone:
                            in_legal_zone = False
                            mask_list[i].append((start_idx, j))
                    else:
                        if not in_legal_zone:
                            in_legal_zone = True
                            start_idx = j
                if in_legal_zone:
                    mask_list[i].append((start_idx, len(sol.actions)))

        frequencies = defaultdict(int) if not self.peek_pos else defaultdict(lambda: [0, defaultdict(int), {}, {}])
        for i, (sol, legal_zones) in enumerate(zip(self.solutions, mask_list)):
            cur_max_abs_len = max_len if max_len is not None else len(sol.actions)
            for start_idx, end_idx in legal_zones:
                last_inds = defaultdict(int)
                for j in range(start_idx, end_idx - min_len + 1):
                    for length in range(min_len, 1 + min(cur_max_abs_len, end_idx - j)):
                        abstract = self.AbsType.from_steps(sol.actions[j:j+length], ex_states=sol.states[j:j+length+1])
                        if j >= last_inds[abstract]:
                            if not self.peek_pos:
                                frequencies[abstract] += 1
                            else:
                                frequencies[abstract][0] += 1
                                wp_abs = AxSeqTreeRelPos.from_steps(sol.actions[j:j+length], ex_states=sol.states[j:j+length+1])
                                frequencies[abstract][1][wp_abs.rel_pos] += 1
                                frequencies[abstract][2][wp_abs.rel_pos] = wp_abs.ex_steps
                                frequencies[abstract][3][wp_abs.rel_pos] = wp_abs.ex_states
                            last_inds[abstract] = j + length
        if not self.peek_pos:
            for abstract in frequencies:
                frequencies[abstract] /= factor
                abstract.freq = frequencies[abstract]
        else:
            for abstract, freq_info in frequencies.items():
                freq_info[0] /= factor
                for rel_pos in freq_info[1]:
                    freq_info[1][rel_pos] /= factor
                abstract.freq, abstract.rel_pos_freq, abstract.rel_pos_ex_steps, abstract.rel_pos_ex_states = freq_info
                frequencies[abstract] = freq_info[0]
        self.frequencies = frequencies
        return frequencies

    def abstract(self):
        """
        Finds common length-2 subsequences (current action, next action)
        that appear with frequency >= thres in dataset of solutions
        """
        thres = (self.num_ax**(-0.75) if self.top is None else 0) if self.thres is None else self.thres  # -0.75 is just an arbitrary exponent
        frequencies = self.frequencies or self.get_frequencies(len(self.solutions))
        pairs = list(filter(lambda x: x.freq >= thres, frequencies))
        pairs.sort(reverse=True, key=lambda x: x.freq)
        top_pairs = pairs[:self.top]
        self.abstractions = top_pairs
        return top_pairs

    def iter_abstract(self, K, get_new_sols=False, num_new_sols=None):
        """
        Abstract common (cur, next) pairs iteratively K times
        """
        sols = self.solutions
        axioms = self.axioms
        for i in range(K):
            abstractor = self.__class__(sols, axioms, self.config)
            abstractions = abstractor.abstract()
            if not abstractions:
                if get_new_sols:
                    return (sols, axioms) if num_new_sols is None else (sols[:num_new_sols], axioms)
                return axioms
            if i == K - 1:
                if get_new_sols:
                    sols = abstractor.abstracted_sol(num_new_sols=num_new_sols)
                    return sols, abstractor.new_axioms
                return axioms + abstractions
            sols = abstractor.abstracted_sol()
            axioms = abstractor.new_axioms


class IAPRandom(IterAbsPairs):
    def __init__(self, solutions, axioms, config):
        super().__init__(solutions, axioms, config)
        self.max_abs_len = config.get("max_abs_len")
        assert self.top is not None

    def abstract(self):
        """
        Returns self.top random subsequences (current action, next action)
        that appear in dataset of solutions
        """
        frequencies = self.frequencies or self.get_frequencies(factor=len(self.solutions), max_len=self.max_abs_len)
        if self.thres is None:
            pairs = list(frequencies)
        else:
            pairs = list(filter(lambda x: x.freq >= self.thres, frequencies))
        top_pairs = random.sample(pairs, self.top)
        self.abstractions = top_pairs
        return top_pairs


class IAPHeuristic(IterAbsPairs):
    def get_frequencies(self, factor=1):
        """
        Gets heuristic scores of (current action, next action) pairs
        Also stores example abstraction for each rel. pos. if --peek
        """
        frequencies = defaultdict(int) if not self.peek_pos else defaultdict(lambda: [0, defaultdict(int), {}, {}])
        for i, sol in enumerate(self.solutions):
            abstracts = []
            for j in range(len(sol.actions)-1):
                abstract = self.AbsType.from_steps(sol.actions[j:j+2], ex_states=sol.states[j:j+3])
                abstracts.append(abstract)
                if len(sol.actions[j]) > 1 or len(sol.actions[j+1]) > 1:
                    abstract_flat = self.AbsType.from_steps(sol.actions[j].steps + sol.actions[j+1].steps, ex_states=sol.actions[j].ex_states[:-1] + sol.actions[j+1].ex_states)
                    abstracts.append(abstract_flat)

            scores = [len(abstract) + abstract.height for abstract in abstracts]

            if not self.peek_pos or self.consider_pos or self.tree_idx:
                for abstract, score in zip(abstracts, scores):
                    frequencies[abstract] += score
            else:
                wp_abstracts = []
                for j in range(len(sol.actions)-1):
                    wp_abs = AxSeqTreeRelPos.from_steps(sol.actions[j:j+2], ex_states=sol.states[j:j+3])
                    wp_abstracts.append(wp_abs)
                    wp_abs_flat = AxSeqTreeRelPos.from_steps(sol.actions[j].steps + sol.actions[j+1].steps, ex_states=sol.actions[j].ex_states[:-1] + sol.actions[j+1].ex_states)
                    wp_abstracts.append(wp_abs_flat)
                for wp_abs, abstract, score in zip(wp_abstracts, abstracts, scores):
                    with_pos, wp_ex_steps, wp_ex_states = wp_abs.rel_pos, wp_abs.ex_steps, wp_abs.ex_states
                    frequencies[abstract][0] += score
                    frequencies[abstract][1][with_pos] += score
                    frequencies[abstract][2][with_pos] = wp_ex_steps
                    frequencies[abstract][3][with_pos] = wp_ex_states
        if not self.peek_pos:
            for abstract in frequencies:
                frequencies[abstract] /= factor
                abstract.freq = frequencies[abstract]
        else:
            for abstract, freq_info in frequencies.items():
                freq_info[0] /= factor
                for rel_pos in freq_info[1]:
                    freq_info[1][rel_pos] /= factor
                abstract.freq, abstract.rel_pos_freq, abstract.rel_pos_ex_steps, abstract.rel_pos_ex_states = freq_info
                frequencies[abstract] = freq_info[0]
        self.frequencies = frequencies
        return frequencies


class IAPLogN(IterAbsPairs):
    def __init__(self, solutions, axioms, config):
        super().__init__(solutions, axioms, config)
        self.max_abs_len = config.get("max_abs_len")
        self.include_eos = config.get("include_eos", False)
        assert self.thres is None

    def get_top_abs(self, mask=None):
        """
        Uses log factor in uniform distribution probabilistic model to determine no. of abs.
        Only chooses one abstraction; use `iter_abstract` to get multiple
        """
        frequencies = self.frequencies or self.get_frequencies(len(self.solutions), self.max_abs_len, mask=mask)
        if not frequencies:
            return None, None
        for abstract in frequencies:
            abstract.score = abstract.freq * (len(abstract) - 1)
        top_abs = max(frequencies, key=lambda ab: ab.score)
        num_abs_ax = len(self.axioms) + self.include_eos
        avg_length = sum(len(sol.actions) for sol in self.solutions) / len(self.solutions) + self.include_eos
        increment = top_abs.score * log(num_abs_ax + 1) - log(1 + 1/num_abs_ax) * avg_length
        self.abstractions = [top_abs]
        return top_abs, increment

    def abstract(self):
        """
        Abstract common (cur, next) pairs iteratively K times
        """
        sols = self.solutions
        axioms = self.axioms
        abstractions = []
        abs_set = set()
        print("ABS SCORE INCREMENTS:")
        for i in itertools.count() if self.top is None else range(self.top):
            abstractor = IAPLogN(sols, axioms, self.config)
            top_abs, increment = abstractor.get_top_abs(mask=abs_set)
            print(top_abs)
            print(increment)
            if increment is None or (self.top is None and increment < 0):
                break
            abstractions.append(top_abs)
            abs_set.add(top_abs)
            sols = abstractor.abstracted_sol()
            axioms = abstractor.new_axioms
        self.abstractions = abstractions
        self.abstracted_solutions = sols
        return abstractions


class IAPEntropy(IterAbsPairs):
    """
    Currently doesn't bias towards higher-frequency abstractions
    """
    def __init__(self, solutions, axioms, config):
        super().__init__(solutions, axioms, config)
        self.ax_freqs = None
        self.max_abs_len = config.get("max_abs_len")
        assert self.top is None and self.thres is None  # Use entropy to determine optimal no. of abs.
        assert not self.peek_pos  # not implemented

    def get_ax_freqs(self, factor=1):
        ax_freqs = defaultdict(int)
        for sol in solutions:
            for action in sol.actions:
                ax_freqs[action.rule] += 1
        if factor != 1:
            for ax in ax_freqs:
                ax_freqs[ax] /= factor
        self.ax_freqs = ax_freqs
        return ax_freqs

    def abstract(self):
        """
        Finds abstractions to decrease entropy
        """
        avg_length = sum(len(sol.states) for sol in self.solutions) / len(self.solutions)  # includes <eos>
        frequencies = self.frequencies or self.get_frequencies(len(self.solutions), self.max_abs_len)
        ax_freqs = self.ax_freqs or self.get_ax_freqs(len(self.solutions))
        # H = -sum(freq * np.log(freq/avg_length) for freq in ax_freqs.values()) + np.log(avg_length)
        top_pairs = []
        for pair, freq in frequencies.items():
            f_ax = [ax_freqs[ax] for ax in pair.rules]
            new_length = avg_length - freq * (len(pair) - 1)
            prob = freq / new_length
            p_ax = list(map(lambda x: x / new_length, f_ax))
            pair.score = freq*log(prob) + sum((fi-freq)*log(pi-prob) for fi, pi in zip(f_ax, p_ax)) \
                - sum(fi*log(pi) for fi, pi in zip(f_ax, p_ax)) \
                + log(avg_length / new_length)
            if pair.score > 0:
                top_pairs.append(pair)
        top_pairs.sort(reverse=True, key=lambda x: x.score)
        self.abstractions = top_pairs
        return top_pairs


class IAPTriePrune(IterAbsPairs):
    def __init__(self, solutions, axioms, config):
        super().__init__(solutions, axioms, config)
        self.max_abs_len = None

    def abstract(self):
        """
        Build trie from common axiom pairs and longer abstractions formed from chaining together pairs
        Take abstractions with high frequency
        Remove abstractions with significantly lower frequency than subabstractions
        """
        top_pairs = set(super().abstract())  # set of Abstraction objects
        abstract_trie = abs_util.Trie()
        for pair in top_pairs:
            abstract_trie.add([(None, pair.rules[0]), (pair.rel_pos[0], pair.rules[1])], 0, (pair.ex_steps, pair.ex_states))
        for sol in self.solutions:
            i = 0
            while i < len(sol.actions) - 1:
                cur_node = abstract_trie.children.get((None, sol.actions[i].rule))
                if cur_node is None:
                    i += 1
                    continue
                j = i
                while j < len(sol.actions) - 1:
                    pair = self.AbsType.from_steps(sol.actions[j:j+2])
                    if pair in top_pairs:
                        next_abs_elt = (pair.rel_pos[0], pair.rules[1])
                        if next_abs_elt in cur_node.children:
                            cur_node = cur_node.children[next_abs_elt]
                            cur_node.value += 1
                        else:
                            cur_node = cur_node.add([next_abs_elt], 1, (sol.actions[i:j+2], sol.states[i:j+3]))
                        j += 1
                    else:
                        break
                i = j + 1
        def dfs_pruner(trie, depth=0):
            if not trie.is_term:
                for sub_trie in trie.children.values():
                    dfs_pruner(sub_trie, depth+1)
            else:
                keys = list(trie.children.keys())
                for key in keys:
                    sub_trie = trie.children[key]
                    if sub_trie.value / trie.value <= (depth - 1) / depth:
                        trie.children.pop(key)
                    else:
                        dfs_pruner(sub_trie, depth+1)
        dfs_pruner(abstract_trie)
        def dfs_make_abs(trie, store_abs, keys_so_far=[]):
            if not trie.children:
                abstraction = self.AbsType.from_abs_elt(keys_so_far, ex_states=trie.comment[1])
                abstraction.ex_steps = trie.comment[0]
                abstraction.freq = trie.value / len(self.solutions)
                store_abs.append(abstraction)
            else:
                for key, sub_trie in trie.children.items():
                    keys_so_far.append(key)
                    dfs_make_abs(sub_trie, store_abs, keys_so_far)
                    keys_so_far.pop()
        abstractions = []
        dfs_make_abs(abstract_trie, abstractions)
        abstractions.sort(key=lambda x: x.freq, reverse=True)
        self.abstractions = abstractions
        return abstractions


class AbsDTrieHeapElt:
    def __init__(self, abstraction: abs_util.DoubleTrie, num_sols: int):
        self.abstraction = abstraction
        self.score = self.abstraction.value * (self.abstraction.depth - 1) / num_sols
        self.removed = False

    def __lt__(self, other):
        return self.score > other.score

from itertools import permutations


class PPO_Learner(Compress):

    def __init__(self, solutions: list[Solution], cur_axioms: list, config: dict):
        super().__init__(solutions, cur_axioms, config)
        self.cur_axioms = cur_axioms
        # self.solutions is inherited from Compress and available here

    class EnvQwer(gym.Env):
        # Define class variables if they are static or initialized in __init__
        # axiom_simplicity, axioms, enc_abs, abs_to_pos might be better as instance variables
        # if they can change per instance or depend on __init__ args.
        # For now, keeping them as they were, but this is a point of attention.

        axiom_simplicity = {
            "refl": 10, "add0": 9, "sub0": 8, "comm": 8, "mul1": 7, "assoc": 7, "div1": 6, "add": 6, "sub": 6, "mul": 6, "div": 6, "zero_div": 5, "dist": 5, "mul0": 4, "sub_comm": 4, "div_self": 3, "subsub": 3, "sub_self": 2, "eval": 1
        }
        # This 'axioms' list seems to be a fixed list of base axiom names,
        # distinct from 'cur_axioms' passed to the EnvQwer constructor.
        base_axioms_names = [
            "refl", "comm", "assoc", "dist", "sub_comm", "eval", "add0", "sub0", "mul1", "div1",
            "div_self", "sub_self", "subsub", "zero_div", "add", "mul0", "sub", "mul", "div"
        ]

        def solution_to_dict(self, solution_obj: Solution) :
            # Assuming Solution and Step objects have __str__ methods
            # and Step has a 'name' attribute or similar for the action string.
            # This is a placeholder based on typical structures.
            # You'll need to adapt it if your Solution/Step structure is different.
            action_sequence = []
            for step in solution_obj.actions: # Assuming solution_obj.actions is a list of Step objects
                action_sequence.append(str(step)) # Or step.name, or however the action is represented
            
            return {
                "problem": str(solution_obj.states[0]), # Assuming first state is the problem
                "solution": action_sequence
            }

        def get_simplicity(self, abstraction_tuple): # Renamed from 'abstraction' to avoid conflict
            # abstraction_tuple is expected to be a tuple of axiom strings (e.g., ('add', 'comm'))
            score = 0
            for ax_name in abstraction_tuple:
                score += self.axiom_simplicity.get(ax_name, 0) # Default to 0 if axiom not in map
            return score

        # Added 'solutions_data' to the constructor
        def __init__(self, cur_axioms_list_of_objects, solutions_data: list[Solution]):
            super().__init__()
            # print("************* PPO_Learner.EnvQwer cur_axioms_list_of_objects:", cur_axioms_list_of_objects)
            
            # Convert Axiom objects to string tuples for internal use if needed
            self.cur_axioms_str_tuples = []
            for ax_obj in cur_axioms_list_of_objects:
                # Assuming Axiom object can be converted to a string tuple representation
                # This might need adjustment based on Axiom's actual structure
                # For example, if ax_obj.rules gives a list of rule names:
                if hasattr(ax_obj, 'rules') and isinstance(ax_obj.rules, (list, tuple)):
                    self.cur_axioms_str_tuples.append(tuple(str(r) for r in ax_obj.rules))
                else: # Fallback to __str__ and splitting, as in original
                    self.cur_axioms_str_tuples.append(tuple(str(ax_obj).split('~')))

            self.enc_abs = [] # List of abstraction tuples (sequences of axiom names)
            self.abs_to_pos = {} # Map from abstraction tuple to its index in enc_abs

            # Populate enc_abs and abs_to_pos using base_axioms_names or cur_axioms_str_tuples
            # The original code used self.axioms, which was the static list.
            # If abstractions should be formed from cur_axioms:
            # source_for_perms = self.cur_axioms_str_tuples # if perms are from current rule set
            # Or, if perms are always from the base set of axiom names:
            source_for_perms = [(name,) for name in self.base_axioms_names] # Treat base axioms as length-1 tuples

            for k in range(1, 3): # Max length of abstractions to consider (e.g., 1 or 2)
                for perm_tuple in permutations(source_for_perms, k):
                    # perm_tuple will be like ((('refl',), ('comm',)),) if k=2 and source_for_perms has tuples
                    # We need to flatten it to a single tuple of strings
                    flattened_perm = []
                    for item_tuple in perm_tuple:
                        flattened_perm.extend(item_tuple)
                    final_perm_tuple = tuple(flattened_perm)
                    
                    if final_perm_tuple not in self.abs_to_pos:
                        self.abs_to_pos[final_perm_tuple] = len(self.enc_abs)
                        self.enc_abs.append(final_perm_tuple)
            
            # print("************* PPO_Learner.EnvQwer self.enc_abs:", self.enc_abs)

            self.observation_space = spaces.MultiBinary(len(self.enc_abs) + 1) # +1 for terminate
            self.action_space = spaces.Discrete(len(self.enc_abs) + 1) # +1 for terminate action

            self.dataset = []
            # Use the passed 'solutions_data'
            if solutions_data:
                for i in range(len(solutions_data)):
                    self.dataset.append(self.solution_to_dict(solutions_data[i]))
            
            self.current_problem_idx = 0
            self.current_actions_mask = np.zeros(len(self.enc_abs) + 1, dtype=np.int8)
            self.current_solution_actions = []
            self.current_step_in_solution = 0

            self.seed() # Initialize random seed

        def seed(self, seed: int = None): # Allow None for random seed
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            return [seed]

        def reset(self):
            if not self.dataset:
                # Handle empty dataset: return a zero observation
                # print("Warning: PPO_Learner.EnvQwer dataset is empty during reset.")
                self.current_actions_mask = np.zeros(len(self.enc_abs) + 1, dtype=np.int8)
                self.current_actions_mask[-1] = 1 # Allow termination
                return self.current_actions_mask.copy()

            self.current_problem_idx = self.np_random.integers(len(self.dataset))
            problem_data = self.dataset[self.current_problem_idx]
            
            # Assuming problem_data["solution"] is a list of action strings/tuples
            # that need to be mapped to indices in self.enc_abs
            self.current_solution_actions = []
            for action_repr in problem_data.get("solution", []):
                # This conversion depends on how actions are stored in solution_to_dict
                # and how they map to self.abs_to_pos keys.
                # Assuming action_repr is a tuple of strings like ('add', 'comm')
                action_tuple = tuple(action_repr) if isinstance(action_repr, list) else action_repr
                if isinstance(action_repr, str): # If it's a single axiom string
                    action_tuple = (action_repr,)

                if action_tuple in self.abs_to_pos:
                    self.current_solution_actions.append(self.abs_to_pos[action_tuple])
                # else:
                    # print(f"Warning: Action {action_tuple} from solution not in abs_to_pos map.")

            self.current_step_in_solution = 0
            
            self.current_actions_mask = np.zeros(len(self.enc_abs) + 1, dtype=np.int8)
            if self.current_step_in_solution < len(self.current_solution_actions):
                action_idx = self.current_solution_actions[self.current_step_in_solution]
                self.current_actions_mask[action_idx] = 1
            self.current_actions_mask[-1] = 1 # Always allow termination
            
            return self.current_actions_mask.copy()

        def get_reward(self, state_mask: np.ndarray, action_idx: int) -> float:
            # state_mask is the current observation (actions taken so far in the episode)
            # action_idx is the ID of the chosen abstraction/action
            
            # If terminate action is chosen
            if action_idx == len(self.enc_abs): # Terminate action
                if self.current_step_in_solution >= len(self.current_solution_actions):
                    return 10.0  # Successfully completed the sequence
                else:
                    return -10.0 # Terminated too early

            # If the chosen action is the correct next action in the solution
            if self.current_step_in_solution < len(self.current_solution_actions) and \
               action_idx == self.current_solution_actions[self.current_step_in_solution]:
                # Reward based on simplicity of the chosen abstraction
                chosen_abstraction_tuple = self.enc_abs[action_idx]
                simplicity_score = self.get_simplicity(chosen_abstraction_tuple)
                return float(simplicity_score)  # Positive reward for correct action
            else:
                return -1.0 # Penalty for incorrect action

        def step(self, action: int):
            # action is the index of the chosen abstraction from self.enc_abs, or terminate action
            
            reward = self.get_reward(self.current_actions_mask, action)
            done = False

            # Update current_actions_mask based on the action taken
            # For PPO, the observation is often the state. Here, the "state"
            # is which actions have been taken to reconstruct the solution.
            # A simple way is to set the bit for the action taken.
            if action < len(self.enc_abs): # Not terminate action
                 # self.current_actions_mask[action] = 1 # Mark action as taken - this might be too simple
                                                     # The observation should reflect the current step in the sequence
                self.current_step_in_solution += 1
            else: # Terminate action
                done = True

            # Prepare next observation
            next_obs_mask = np.zeros(len(self.enc_abs) + 1, dtype=np.int8)
            if not done and self.current_step_in_solution < len(self.current_solution_actions):
                next_action_idx = self.current_solution_actions[self.current_step_in_solution]
                next_obs_mask[next_action_idx] = 1 # Highlight the next correct action
            
            if self.current_step_in_solution >= len(self.current_solution_actions) and not done:
                # If all solution steps are done, but not terminated, force termination observation
                done = True # Or allow only termination

            next_obs_mask[-1] = 1 # Always allow termination in the next step
            
            self.current_actions_mask = next_obs_mask # Update for the next call to get_reward if needed

            return self.current_actions_mask.copy(), reward, done, {} # {} is info dict

    def iter_abstract(self, num_total_timesteps): # Renamed 'iter' to 'num_total_timesteps'
        if not self.solutions:
            print("PPO_Learner: No solutions provided to learn from. Skipping PPO training.")
            return self.cur_axioms # Return current axioms if no solutions

        # Pass self.solutions to EnvQwer
        env = self.EnvQwer(self.cur_axioms, self.solutions)

        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            verbose=2
        )
        
        # Check if environment has any encodable abstractions
        if not env.enc_abs:
            print("PPO_Learner.EnvQwer has no encodable abstractions (env.enc_abs is empty). Skipping PPO training.")
            return self.cur_axioms


        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            n_steps=1024, # Or make configurable
            batch_size=64, # Or make configurable
            n_epochs=10,   # Or make configurable
            learning_rate=1e-4, # Or make configurable
            clip_range=0.1,     # Or make configurable
            device="cuda" if torch.cuda.is_available() else "cpu", # Check for cuda
        )

        model.learn(total_timesteps=num_total_timesteps, callback=wandb_callback)
        # model.learn(total_timesteps=num_total_timesteps)
        
        # After learning, determine which abstractions were chosen/reinforced.
        # This part is crucial and needs a strategy.
        # For example, analyze the PPO policy or observe its behavior on the training data.
        # The original code had a simple rollout and collected chosen actions.

        # Simple rollout to see what the agent picks:
        # This is a placeholder for a more sophisticated way to extract "learned" abstractions.
        
        learned_abstractions_indices = set()
        if env.dataset: # Only attempt rollout if there's data
            for _ in range(min(len(env.dataset), 10)): # Rollout on a few problems
                obs = env.reset()
                done = False
                ep_actions = []
                while not done:
                    action_indices, _states = model.predict(obs, deterministic=True)
                    # model.predict might return a single action index or a list if env is vectorized
                    action_idx = action_indices.item() if isinstance(action_indices, np.ndarray) else action_indices
                    
                    if action_idx < len(env.enc_abs): # If not terminate action
                        ep_actions.append(action_idx)
                    obs, reward, done, info = env.step(action_idx)
                learned_abstractions_indices.update(ep_actions)
        
        # Convert chosen indices back to Abstraction/Axiom objects
        # The new abstractions should be added to self.cur_axioms
        # This requires knowing the AbsType for creating new Abstraction objects.
        # self.AbsType is available from the parent Compress class.

        final_abstractions_objects = list(self.cur_axioms) # Start with existing axioms
        
        # print(f"PPO Learned abstraction indices: {learned_abstractions_indices}")
        # print(f"PPO Env enc_abs: {env.enc_abs}")

        for idx in learned_abstractions_indices:
            if 0 <= idx < len(env.enc_abs):
                abs_tuple = env.enc_abs[idx] # This is a tuple of strings, e.g., ('add', 'comm')
                # Create a new Abstraction object.
                # This assumes self.AbsType can be initialized from a sequence of rule names.
                try:
                    # The constructor for Abstraction might vary.
                    # If it takes a list of rule objects or rule names:
                    # Option 1: If AbsType takes rule names directly
                    new_abs_obj = self.AbsType(rules=list(abs_tuple)) 
                    # Option 2: If AbsType needs Axiom objects for each rule name
                    # rule_objects = [Axiom(name) for name in abs_tuple] # Assuming Axiom can be made from name
                    # new_abs_obj = self.AbsType(rules=rule_objects)
                    
                    # Add score or frequency if your Abstraction objects store them
                    # new_abs_obj.score = env.get_simplicity(abs_tuple) # Example
                    
                    # Avoid adding duplicates if they are already in cur_axioms or added
                    is_duplicate = False
                    for existing_ax in final_abstractions_objects:
                        # This comparison logic depends heavily on how Axiom/Abstraction equality is defined.
                        # If they have a 'rules' attribute that is a list/tuple of strings:
                        if hasattr(existing_ax, 'rules') and tuple(str(r) for r in existing_ax.rules) == abs_tuple:
                            is_duplicate = True
                            break
                        # Fallback to string comparison if rules attribute isn't standard
                        elif str(existing_ax) == "~".join(abs_tuple): # Heuristic
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        final_abstractions_objects.append(new_abs_obj)
                except Exception as e:
                    print(f"Error creating/adding abstraction object for {abs_tuple}: {e}")


        # print(f"PPO Final abstractions: {[str(a) for a in final_abstractions_objects]}")
        return final_abstractions_objects




class IAPDTrieLogN(IAPLogN):
    def get_frequencies(self, factor=1, max_len=None):
        frequencies = abs_util.DoubleTrie(accum=True)
        for sol in self.solutions:
            if len(sol.actions) > 1:
                abstraction = self.AbsType.from_steps(sol.actions)
                frequencies.add(abstraction.rules, 1, abstraction.__dict__.get("rel_pos"), max_length=max_len)
        if factor != 1:
            def dfs(node, not_root=True):
                if not_root:
                    node.value /= factor
                for child in node.append_children.values():
                    dfs(child)
            dfs(frequencies, False)
        self.frequencies = frequencies
        return frequencies

    def abstract(self):
        frequencies = self.frequencies or self.get_frequencies()

        def dfs(node, depth=0):
            if depth >= 2:
                yield node
            if self.max_abs_len is None or depth < self.max_abs_len:
                for child in node.append_children.values():
                    yield from dfs(child, depth+1)
        abs_heap = [AbsDTrieHeapElt(ab, len(self.solutions)) for ab in dfs(frequencies)]
        heapq.heapify(abs_heap)
        abs_heap_dict = {abs_heap_elt.abstraction: abs_heap_elt for abs_heap_elt in abs_heap}

        def pop_heap():
            while (top_heap_elt := heapq.heappop(abs_heap)).removed:
                continue
            top_node = top_heap_elt.abstraction
            del abs_heap_dict[top_node]
            return top_node, top_heap_elt.score

        def update(node):
            if node in abs_heap_dict:
                abs_heap_dict[node].removed = True
                new_heap_elt = AbsDTrieHeapElt(node, len(self.solutions))
                heapq.heappush(abs_heap, new_heap_elt)
                abs_heap_dict[node] = new_heap_elt

        def dfs_overlap(node, super_node, prepend: bool):  # l > L, r >= R; l <= L, r < R
            node.value -= super_node.value
            update(node)
            children = node.prepend_children if prepend else node.append_children
            super_children = super_node.prepend_children if prepend else super_node.append_children
            for next_key, next_node in children.items():
                if next_node.value and next_key in super_children:
                    dfs_overlap(next_node, super_children[next_key], prepend)

        def dfs_super_helper(node):
            node.value = 0
            update(node)
            for next_node in node.append_children.values():
                if next_node.value:
                    dfs_super_helper(next_node)

        def dfs_super(node):  # l <= L, r >= R
            dfs_super_helper(node)
            for next_node in node.prepend_children.values():
                if next_node.value:
                    dfs_super(next_node)

        num_abs_ax = len(self.axioms) + self.include_eos
        avg_length = sum(len(sol.actions) for sol in self.solutions) / len(self.solutions) + self.include_eos
        top_abs = []
        print("ABS SCORE INCREMENTS:")
        for count in itertools.count() if self.top is None else range(self.top):
            top, score = pop_heap()
            occurs = top.value
            increment = score * log(num_abs_ax + 1) - log(1 + 1/num_abs_ax) * avg_length
            print(increment)
            if self.top is None and increment < 0:
                break
            top_abs.append((top, occurs / len(self.solutions), score))
            num_abs_ax += 1
            avg_length -= score
            # l > L, r >= R; l <= L, r < R
            for prepend in [False, True]:
                cur_node = top
                for _ in range(top.depth-1):
                    cur_node = cur_node.append_parent if prepend else cur_node.prepend_parent
                    dfs_overlap(cur_node, top, prepend)
            # l <= L, r >= R
            dfs_super(top)
            # l > L, r < R
            node = top
            for i in range(1, top.depth-1):
                node = node.prepend_parent
                cur_node = node
                for _ in range(1, top.depth-i):
                    cur_node = cur_node.append_parent
                    cur_node.value -= occurs
                    update(cur_node)

        def get_shifted_abs_elts(node):
            cur_node = node
            for _ in range(node.depth):
                yield cur_node.prepend_key
                cur_node = cur_node.prepend_parent
        self.abstractions = [self.AbsType.from_shifted_abs_elt(get_shifted_abs_elts(node)) for node, _, _ in top_abs]
        for ab, (_, freq, score) in zip(self.abstractions, top_abs):
            ab.freq = freq
            ab.score = score
        return self.abstractions


COMPRESSORS = {"iap": IterAbsPairs, "iap_rand": IAPRandom, "iap_heur": IAPHeuristic,
               "iap_ent": IAPEntropy, "iap_logn": IAPLogN, "iap_trie": IAPTriePrune, "iap_dtrie": IAPDTrieLogN, "ppo":PPO_Learner}

def debug():
    """ Add code here for debugging """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find mathematical absractions.")
    parser.add_argument("-t", dest="test", action="store_true", help="Testing")
    parser.add_argument("--config", dest="config_file", type=str, help="Configuration file (can override other options)")
    parser.add_argument("--sol-data", type=str, help="Path to data of solutions from which abstractions are found")
    parser.add_argument("--file", type=str, help="File to store the abstractions")
    parser.add_argument("--sol-file", type=str, help="File to store the abstracted solutions")
    parser.add_argument("--use", dest="num_use", type=int, default=None, help="How many solutions to use (default: all)")
    parser.add_argument("--store", dest="num_store", type=int, default=None, help="How many abstracted solutions to store (default: all)")
    parser.add_argument("--num-ex-sol", dest="num_ex_sol", type=int, default=0, help="Number of example solutions to display")
    parser.add_argument("-s", dest="small", action="store_true", help="Whether to use small dataset")
    parser.add_argument("--iter", type=int, default=1, help="How many times to iterate pair abstraction process")
    parser.add_argument("--thres", type=float, default=None, help="Threshold frequency for abstractions")
    parser.add_argument("--top", metavar="K", type=int, default=None, help="Choose top K abstractions")
    parser.add_argument("--abs-type", dest="abs_type", default="tree_rel_pos", choices=["ax_seq", "dfs_idx_rel_pos", "tree_rel_pos"], help="Type of abstraction")
    parser.add_argument("--compressor", default="iap_heur", choices=COMPRESSORS, help="Type of algorithm for discovering abstractions")
    parser.add_argument("--peek", dest="peek_pos", action="store_true", help="Take peek at relative positions (with abs. type tree_rel_pos) even when we don't consider them")
    parser.add_argument("-v", dest="verbose", action="store_true", help="Display example axioms")
    parser.add_argument("--debug", action="store_true", help="Debug")

    args = parser.parse_args()
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            args.__dict__.update(json.load(f))

    if args.debug:
        debug()
        sys.exit()

    if args.small:
        num_use = args.num_use or 8000
        num_store = args.num_store or 8000
    else:
        num_use = args.num_use or 80000
        num_store = args.num_store or 80000

    if args.sol_data is None:
        if args.abs_type == "tree_rel_pos":
            solutions = abs_util.load_solutions("data/equations-8k.json")
        else:
            if args.small:
                solutions = abs_util.load_solutions("data/equations-8k.json")
            else:
                solutions = abs_util.load_solutions("data/equations-8k.json")
    elif isinstance(args.sol_data, str):
        with open(args.sol_data, 'r') as f:
            solutions = abs_util.load_solutions(args.sol_data)
    else:
        solutions = args.sol_data
    _, axioms = abs_util.load_axioms("equation_axioms.json")
    axioms = [Axiom(ax_name) for ax_name in axioms]

    if args.test:
        doctest.testmod()
    else:
        used_sols = solutions[:num_use] if isinstance(num_use, int) else [solutions[i] for i in num_use]
        solutions = [Solution.from_dict(sol, AbsType=ABS_TYPES[args.abs_type]) for sol in used_sols]
        start_time = datetime.now()
        compressor = COMPRESSORS[args.compressor](solutions, axioms, vars(args))
        if args.sol_file is not None or args.num_ex_sol:
            sols, abs_ax = compressor.iter_abstract(args.iter, True, num_store)
            print("TOTAL TIME:", datetime.now() - start_time)
            if args.sol_file is not None:
                with open(args.sol_file, "wb") as f:
                    pickle.dump(sols, f)
            if args.num_ex_sol is not None:
                for i, ex_sol in enumerate(random.sample(sols, args.num_ex_sol)):
                    print(f"EXAMPLE SOLUTION {i+1}")
                    print(ex_sol)
                    print('\n')
        else:
            abs_ax = compressor.iter_abstract(args.iter)
            print("TOTAL TIME:", datetime.now() - start_time)
        if args.file is not None:
            if args.file[-5:] == ".json":
                abs_ax_str = list(map(str, abs_ax))
                with open(args.file, "w") as f:
                    json.dump({"num": len(abs_ax_str), "axioms": abs_ax_str}, f)
            elif args.file[-4:] == ".pkl":
                with open(args.file, "wb") as f:
                    pickle.dump(abs_ax, f)
        print("****************-",abs_ax)
        print("****************-",axioms)
        print("NUM NEW ABS:", len(abs_ax) - len(axioms))
        if args.abs_type in ["tree_rel_pos", "dfs_idx_rel_pos"] or not args.peek_pos:
            for i in range(len(axioms), len(abs_ax)):
                print(f"{str(abs_ax[i])}\n\t{abs_ax[i].score or abs_ax[i].freq}")
                if args.verbose:
                    print('\tEx.  ')
                    for j in range(len(abs_ax[i].ex_steps)):
                        print(f"\t\t{abs_ax[i].ex_states[j]}")
                        print(f"\t\t\t{abs_ax[i].ex_steps[j]}")
                    print(f"\t\t{abs_ax[i].ex_states[-1]}")
        else:
            for i in range(len(axioms), len(abs_ax)):
                print(str(abs_ax[i]))
                sorted_rel_pos = sorted(((freq, rp) for rp, freq in abs_ax[i].rel_pos_freq.items()), reverse=True)
                for freq, rp in sorted_rel_pos:
                    print(f"\t{', '.join(map(str, rp))}\n\t\t{freq}")
                    if args.verbose:
                        print('\t\tEx.  ')
                        for j in range(len(abs_ax[i].rel_pos_ex_steps[rp])):
                            print(f"\t\t\t{abs_ax[i].rel_pos_ex_states[rp][j]}")
                            print(f"\t\t\t\t{abs_ax[i].rel_pos_ex_steps[rp][j]}")
                        print(f"\t\t\t{abs_ax[i].rel_pos_ex_states[rp][-1]}")
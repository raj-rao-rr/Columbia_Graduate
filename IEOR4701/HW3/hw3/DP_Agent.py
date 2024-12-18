"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A dynamic programming agent for a stochastic task environment
"""

import random
import math
import sys


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])

class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None

    
    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass


    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass


    def value_iteration(self, valid_actions, transition):
        """ Computes all optimal values using value iteration and stores them in self.values.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        # TODO
        
        error = math.inf
        while error > 1e-6:
            old_values = self.values.copy()
            
            for s in self.states:
                possible_actions = valid_actions(s)
                
                # assuming a transiiton probability of 1.0 
                self.values[s] = max([transition(s, a)[1] + self.gamma*self.values[transition(s, a)[0]] if a is not None else 0 for a in possible_actions])
                
            error = sum([abs(old_values[s] - self.values[s]) for s in self.states])

    def policy_extraction(self, valid_actions, transition):
        """ Computes all optimal actions using value iteration and stores them in self.policy.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        # TODO
        
        for s in self.states:
            possible_actions = valid_actions(s)
            optimal_action = argmax([transition(s, a)[1] + self.gamma*self.values[transition(s, a)[0]] for a in possible_actions])
            self.policy[s] = possible_actions[optimal_action]
            

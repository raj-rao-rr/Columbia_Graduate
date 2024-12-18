#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
MCTS AI player for Othello.
"""

import random
import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility


class Node:
    def __init__(self, state, player, parent, children, v=0, N=0):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = children
        self.value = v
        self.N = N

    def get_child(self, state):
        for c in self.children:
            if (state == c.state).all():
                return c
        return None


def uct_func(node, alpha):
    """"Compute the UCT value to use in selection"""
    
    exploitation = (node.value / node.N)
    
    if node.parent == None:
        exploration = alpha * np.sqrt( np.log(0) / node.N )
    else:
        exploration = alpha * np.sqrt( np.log(node.parent.N) / node.N )
    
    UCT = exploitation + exploration
    
    return UCT


def select(root, alpha):
    
    """ Starting from given node, find a terminal node or node with unexpanded children.
    If all children of a node are in tree, move to the one with the highest UCT value.

    Args:
        root (Node): MCTS tree root node
        alpha (float): Weight of exploration term in UCT

    Returns:
        node (Node): Node at bottom of MCTS tree
    """
    # TODO:
    
        # children list of Nodes (revise the np.isin)
        # start at root node, check to see if children haven't spawned if they haven't spawend (then return node) same if no children then return,
        # if you've seen all children then move to highest ucts
        # i.stae for i in root.children 
    
    successor = get_possible_moves(root.state, root.player)       
    
    if (len(successor) == 0) or \
        (np.isin(successor, [i.state for i in root.children]).sum() > 0):
        return root
    
    else:     
        
        local_max, local_node = -float("inf"), None
        
        # should be iterating through children node
        for action in successor:
            
            # reconsider how UCT is being called
            uct_value = uct_func(root, alpha)
            
            if uct_value > local_max:
                local_max, local_node = uct_value, action
        
        return select(local_node, alpha) 


def expand(node):
    """ Add a child node of state into the tree if it's not terminal.

    Args:
        node (Node): Node to expand

    Returns:
        leaf (Node): Newly created node (or given Node if already leaf)
    """
    # TODO:
    
    
    return node


def simulate(node):
    """ Run one game rollout using from state to a terminal state.
    Use random playout policy.

    Args:
        node (Node): Leaf node from which to start rollout.

    Returns:
        utility (int): Utility of final state
    """
    # TODO:  
    
    # move to terminal node and report back figure
    if len(get_possible_moves(node.state, node.player)) == 0:
        return compute_utility(node.state)
    else:
        moves = get_possible_moves(node.state, node.player)    
        action = random.choice(moves)
        state_space = play_move(node.state, node.player, action[0], action[1])
        
        return Node(state_space, node.player, node, [], 0, node.N)


def backprop(node, utility):
    """ Backpropagate result from state up to the root.
    Every node has N, number of plays, incremented
    If node's parent is dark (1), then node's value increases
    Otherwise, node's value decreases.

    Args:
        node (Node): Leaf node from which rollout started.
        utility (int): Utility of simulated rollout.
    """
    # TODO:
    
    if node.parent == None:
        node.N += 1
        if node.player == 1:
            node.value = -utility 
            utility += 1
        else:
            node.value = utility 
            utility -= 1
            
    else:
        
        node.N += 1
        if node.player == 1:
            node.value = -utility 
            utility += 1
        else:
            node.value = utility 
            utility -= 1
            
        return backprop(node.parent, utility)


def mcts(state, player, rollouts=100, alpha=5):
    # MCTS main loop: Execute four steps rollouts number of times
    # Then return successor with highest number of rollouts
    root = Node(state, player, None, [], 0, 1)
    for i in range(rollouts):
        leaf = select(root, alpha)
        new = expand(leaf)
        utility = simulate(new)
        backprop(new, utility)

    move = None
    plays = 0
    for m in get_possible_moves(state, player):
        s = play_move(state, player, m[0], m[1])
        if root.get_child(s).N > plays:
            plays = root.get_child(s).N
            move = m

    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("MCTS AI")        # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = mcts(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
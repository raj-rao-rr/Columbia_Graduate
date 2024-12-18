#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Alpha-beta minimax AI player for Othello.
"""

import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility


def max_value(state, player, alpha, beta):
    """
    Args:
        state: Board state
        player: Dark (1) or light (2)
        alpha, beta values

    Returns:
        value (int): Minimax value of state
        move (tuple): Best move to make
    """
    # TODO
    
    # we've reached a terminal node, return it's value
    successor = get_possible_moves(state, player)
    if len(successor) == 0:
        return compute_utility(state), None
    
    v = -float("inf")
    
    # iterate through all available states and check min value
    for action in successor:
        
        if player == 1:
            v2, a2 = min_value(play_move(state, player, action[0], action[1]),
                               2, alpha, beta)
        else:
            v2, a2 = min_value(play_move(state, player, action[0], action[1]), 
                               1, alpha, beta)
            
        if v2 > v:
            v, move = v2, action
            alpha = max(alpha, v)
        
        if v >= beta:
            return v, move
    
    return v, move
                

def min_value(state, player, alpha, beta):
    """
    Args:
        state: Board state
        player: Dark (1) or light (2)
        alpha, beta values

    Returns:
        value (int): Minimax value of state
        move (tuple): Best move to make
    """
    # TODO
    
    # we've reached a terminal node, return it's value
    successor = get_possible_moves(state, player)
    if len(successor) == 0:
        return compute_utility(state), None
    
    v = float("inf")
    
    # iterate through all available states and check min value
    for action in successor:
        
        if player == 1:
            v2, a2 = max_value(play_move(state, player, action[0], action[1]), 
                               2, alpha, beta)
        else:
            v2, a2 = max_value(play_move(state, player, action[0], action[1]), 
                               1, alpha, beta)
        
        if v2 < v:
            v, move = v2, action
            beta = min(beta, v)
        
        if v <= alpha:
            return v, move
    
    return v, move


def minimax(state, player):
    """
    Minimax main loop
    Call max_value if player is 1 (dark), min_value if player is 2 (light)
    Then return the resultant move
    """
    if player == 1:
        _, move = max_value(state, player, -float('inf'), float('inf'))
    else:
        _, move = min_value(state, player, -float('inf'), float('inf'))
    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Minimax AI")     # First line is the name of this AI
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
            movei, movej = minimax(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
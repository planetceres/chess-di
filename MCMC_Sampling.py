
# coding: utf-8

import numpy as np
import sys

# Construct chess board
chess_board = np.reshape(np.linspace(dtype=int, num=16, start=0, stop=15), (4,4))
# Constrain moves to chess board
chess_board_index = [i for i in np.ndindex(4,4)]

# Define action space for knight
def knight_action(x, y):
    # Create empty array for valid moves
    valid_actions = np.zeros(shape=(4,4), dtype=int)
    # Define possible moves
    action_space = [(x+1, y+2), (x+1, y-2),
                    (x-1, y+2), (x-1, y-2),
                    (x+2, y+1), (x+2, y-1),
                    (x-2, y-1), (x-2, y+1)]
    # Mark current location
    valid_actions[(x,y)] = -1

    # Report valid moves based on current location
    for i in action_space:
        if i in chess_board_index:
            valid_actions[i] = 1

    # Report indices of valid moves
    valid_indices = np.dstack(np.where(valid_actions == 1))[0]

    return valid_actions, valid_indices

# Define action space for knight
def knight_action(x, y):
    # Create empty array for valid moves
    valid_actions = np.zeros(shape=(4,4), dtype=int)
    # Define possible moves
    action_space = [(x+1, y+2), (x+1, y-2),
                    (x-1, y+2), (x-1, y-2),
                    (x+2, y+1), (x+2, y-1),
                    (x-2, y-1), (x-2, y+1)]
    # Mark current location
    valid_actions[(x,y)] = -1

    # Report valid moves based on current location
    for i in chess_board_index:
        if i in action_space:
            valid_actions[i] = 1

    # Report indices of valid moves
    valid_indices = np.dstack(np.where(valid_actions == 1))[0]

    return valid_actions, valid_indices


# Select an action with uniform probability based on current location
def action_select(position):
    # Get valid indices for next move
    _, valid_indices = knight_action(position[0], position[1])
    # Choose random action
    random_action = np.random.choice(valid_indices.shape[0])
    # Indices of next move
    x, y = valid_indices[random_action][0], valid_indices[random_action][1]

    return x, y


# Get value of key on chess board at position (x,y)
def get_key(position):
    return chess_board[position]


def get_score(trials, print_stat=False):
    total_score = 0
    position = (0,0)
    for i in range(trials):
        trial = i + 1
        position = action_select(position)
        score = get_key(position)
        total_score += score
        if print_stat:
            print_string = (trial, position, score, total_score)
            print("{}\n".format(print_string))
            valid_actions = np.zeros(shape=(4,4), dtype=int)
            valid_actions[position] = -1
            print("{}\n".format(valid_actions))
    return total_score


def get_modulo_value(moves, modulo, print_stat=False):
    running_sum, modulos  = [], []
    for i in range(1,(moves+1)):
        score = get_score(moves, print_stat=False)

        # Add current score to running sum
        if len(running_sum) > 0:
            sum_with_prev = score + running_sum[-1]
        else:
            sum_with_prev = score

        # store modulo values
        score_modulo = sum_with_prev % modulo
        running_sum.append(sum_with_prev)
        modulos.append(score_modulo)

        if print_stat: print(i, moves, score, score_modulo, running_sum, modulos)
    return running_sum, modulos


# Run sampling inference
def sample_modulo_results(moves, modulo, samples=1000, print_stat=False):
    means, stds = [], []
    for i in range(samples):
        _, results = get_modulo_value(moves, modulo, print_stat=print_stat)
        modulo_mu = np.mean(results)
        modulo_std = np.std(results)
        means.append(modulo_mu)
        stds.append(modulo_std)
        if i % 100 == 0:
            print("{}/{} iterations complete | mean : {} std: {}".format(i, samples, np.mean(means), np.std(stds)))
    return np.mean(means), np.std(stds)


# Conditional on two divisors
def conditional_sample_modulo_results(moves, modulo_1, modulo_2, samples=1000, print_stat=False):
    total_1, total_2 = 0, 0
    for i in range(samples):
        running_sum, _ = get_modulo_value(moves, modulo_1, print_stat=print_stat)
        running_sum = running_sum[-1]

        # If the sum is divisible by first number, check if is it also divisible be the second
        if running_sum % modulo_1 == 0:
            total_1 += 1
            if running_sum % modulo_2 == 0:
                total_2 += 1
        # Logging for progress
        if i > 1 and (i % 1000 == 0):
            print("{}/{} iterations complete | total_1 : {} total_2: {}".format(i, samples, total_2, total_1))

    # Probability that sum is divisible by second number, given that it is divisble by first
    probability = 1.0*total_2/total_1
    print("{}/{} probability {}".format(total_2, total_1, probability))
    return probability


# Moves: 16, Divisible by 13
sample_modulo_results(16, 13, samples=100000, print_stat=False)
# Moves: 512, Divisible by 311
sample_modulo_results(512, 311, samples=100, print_stat=False)
# Moves: 16, Divisible by 13 and also 5
conditional_sample_modulo_results(16, 13, 5, samples=10000, print_stat=False)
# Moves: 512, Divisible by 43 and also 7
conditional_sample_modulo_results(512, 43, 7, samples=100, print_stat=False)

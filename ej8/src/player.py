import sys
from random import sample
import numpy as np
from sklearn import preprocessing
from src.util import as_vector

class Player:
    initial_weights = [1, 5, -5, -5, 0.5]

    def __init__(self, target, marbles, color, board, weights=initial_weights):
        self.target = target
        self.marbles = marbles
        self.color = color
        self.board = board
        self.rate = 0.0000001
        self.weights = weights
        self.next_value = None
        self.moves = 0
        self.match_errors = []

    def __str__(self):
        return f'Target: {self.target} \nMarbles: {[str(marble) for marble in self.marbles]} \nColor: {self.color}'

    def possible_moves(self):
        possible_moves = set()
        for marble in self.marbles:
            marble_moves = { (marble, dst) for dst in  marble.possible_moves() }
            possible_moves = possible_moves | marble_moves
        return possible_moves

    def random_move(self):
        possible_moves = self.possible_moves()
        if len(possible_moves) == 0:
            return False

        marble, move = sample(possible_moves, 1)[0]

        marble.make_move(move)
        self.moves += 1

        return True, None

    def best_move(self, learn=True):
        # If player learns, and is not in its first turn
        if learn and self.next_value:
            self.adjust_weights()

        best_value = float('-inf')
        next_move = None
        marble_to_move = None
        for marble, dst in self.possible_moves():
            next_board = marble.make_move(dst, calculate_move=True)
            args = as_vector(next_board, self.color)
            next_value = self.learning_function(args)

            if next_value > best_value:
                best_value = next_value
                next_move = dst
                marble_to_move = marble
                self.next_value = next_value

        if marble_to_move is None:
            return False, self.match_errors

        marble_to_move.make_move(next_move)
        self.moves += 1

        return True, self.match_errors

    def adjust_weights(self):
        expected_value = self.next_value
        actual_board = (1,) +  as_vector(self.board, self.color)
        actual_value = self.learning_function(as_vector(self.board, self.color))
        new_weights = []
        error = (expected_value - actual_value) ** 2
        self.match_errors.append(error)

        for i, weight in enumerate(self.weights):
            new_weights.append(weight + self.rate * (expected_value - actual_value) * actual_board[i])
        self.weights = new_weights

    def learning_function(self, args):
        in_target = args[0]
        oponent_in_target = args[1]
        my_total_distance_to_zone = args[2]
        op_total_distance_to_zone = args[3]

        if in_target == 10:
            return 1000
        elif oponent_in_target == 10:
            return -1000

        w0, w1, w2, w3, w4 = self.weights
        result = w0 + w1*in_target + w2*oponent_in_target +  w3*my_total_distance_to_zone + w4*op_total_distance_to_zone
        return result

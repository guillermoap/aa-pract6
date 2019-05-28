from src.player import Player
from src.neural import NNQFunction
from src.util import as_vector
from random import sample
from copy import deepcopy
import numpy as np

class NeuralPlayer:
    initial_weights = [1, 5, -5, -5, 0.5]

    def __init__(self, target, marbles, color, board, weights):
        self.target = target
        self.marbles = marbles
        self.color = color
        self.board = board
        self.moves = 0
        self.match_errors = []
        self.weights = weights
        self.adjust_rate = 2
        self.rate = self.weights.rate
    
    def __deepcopy__(self, memo):
        return NeuralPlayer(self.target, deepcopy(self.marbles, memo), self.color,  deepcopy(self.board, memo), self.weights)

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
        if learn:
            print('Adjusting Weights')
            if (self.moves + 1) % self.adjust_rate == 0:
                self.adjust_weights()
            vector = as_vector(self.board, self.color)
            self.weights.add_state(vector, self.weights.value(vector))
        
        if self.weights.should_explore():
            return self.random_move()
        
        best_value = float('-inf')
        next_move = None
        marble_to_move = None
        for marble, dst in self.possible_moves():
            next_board = marble.make_move(dst, calculate_move=True)
            next_value = self.learning_function(as_vector(next_board, self.color))

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
        self.weights.adjust()
        self.weights.clear_states()

    def learning_function(self, state):
        return self.weights.value(state)

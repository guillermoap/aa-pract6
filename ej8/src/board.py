import numpy as np
from colorama import Fore
from src.marble import Marble
from src.consts import *
from src.neural_player import NeuralPlayer
from src.player import Player
from src.neural import NNQFunction
from src.util import initial_positions, oponent_color
'''
    The board is inspired by https://github.com/mavlee/chinese-checkers/blob/master/ChineseCheckersV4.java
    Allowed movements are right, left and diagonal (up and down would allow extramovents)
'''
class Board:
    def __init__(self, weights=None, types=[NEURAL_TYPE, NEURAL_TYPE]):
        self.clear()
        self.weights = weights
        self.init_player(RED, types[1])
        self.init_player(GREEN, types[0])

    def __str__(self):
        s = ''
        for h in range(BOARD_HEIGHT):
            for w in range(BOARD_WIDTH):
                x = self.board[h][w]
                if x == RED:
                    s += f'{Fore.RED}{x}'
                elif x == GREEN:
                    s += f'{Fore.GREEN}{x}' if x != -1 else ' '
                else:
                    s += f'{Fore.WHITE}{x}' if x != -1 else ' '
            s += '\n'
        return s


    '''
        Clears the current board, or initialize a new one
    '''
    def clear(self):
        self.players = dict()
        self.board = np.full((BOARD_HEIGHT, BOARD_WIDTH), INVALID_SPACE)

        for h in range(BOARD_HEIGHT):
            if h == 0 or h == 16:
                self.board[h][12] = EMPTY_SPACE
            elif h == 2 or h == 14:
                for w in range(10,15,2):
                    self.board[h][w] = EMPTY_SPACE
            elif h == 4 or h == 12:
                for w in range(0,25,2):
                    self.board[h][w] = EMPTY_SPACE
            elif h == 6 or h == 10:
                for w in range(2,23,2):
                    self.board[h][w] = EMPTY_SPACE
            elif h == 8:
                for w in range(4,21,2):
                    self.board[h][w] = EMPTY_SPACE
            elif h == 1 or h == 15:
                for w in range(11,14,2):
                    self.board[h][w] = EMPTY_SPACE
            elif h == 3 or h == 13:
                for w in range(9,16,2):
                    self.board[h][w] = EMPTY_SPACE
            elif h == 5 or h == 11:
                for w in range(1,24,2):
                    self.board[h][w] = EMPTY_SPACE
            elif h == 7 or h == 9:
                for w in range(3,22,2):
                    self.board[h][w] = EMPTY_SPACE
    '''
        Initialize players' marbles.
        Currently red and green are the only supported colors.
    '''
    def init_player(self, color, player_type):
        marbles = set()
        target = set()
        # target zone
        target = initial_positions(oponent_color(color)) # modulo 6, but starting at 1 instead of 0
        # initial zone
        for position in initial_positions(color):
            i,j = position
            self.board[i][j] = color
            marbles.add(Marble(self, position, color, target))
        if color == GREEN:
            if self.weights[0] is not None:
                if player_type == NEURAL_TYPE:
                    self.players[color] = NeuralPlayer(target, marbles, color, self, self.weights[0])  
                else:
                    self.players[color] = Player(target, marbles, color, self, self.weights[0]) 
            else:
                if player_type == NEURAL_TYPE:
                    self.players[color] = NeuralPlayer(target, marbles, color, self, NNQFunction(color))
                else:
                    self.players[color] = Player(target, marbles, color, self)
        elif color == RED:
            if self.weights[1] is not None:
                if player_type == NEURAL_TYPE:
                    self.players[color] = NeuralPlayer(target, marbles, color, self, self.weights[1])
                else:
                    self.players[color] =  Player(target, marbles, color, self, self.weights[1])
            else:
                if player_type == NEURAL_TYPE:
                    self.players[color] = NeuralPlayer(target, marbles, color, self, NNQFunction(color))
                else:
                    self.players[color] = Player(target, marbles, color, self)


    def is_valid(self, position):
        i, j = position
        return (0 <= i < BOARD_HEIGHT) and (0 < j < BOARD_WIDTH) and self.board[i, j] != INVALID_SPACE

    def is_ocuppied(self, position):
        i, j = position
        return self.board[i, j] != EMPTY_SPACE and self.board[i, j] != INVALID_SPACE

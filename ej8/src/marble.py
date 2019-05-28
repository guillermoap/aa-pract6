from copy import deepcopy
from math import sqrt
from src.consts import EMPTY_SPACE, MOVES, BOARD_HEIGHT, RED
import random

class Marble:
    def __init__(self, board, position, color, target):
        self.board = board
        self.position = position
        self.previous_position = None
        self.color = color
        self.target = target

    def __str__(self):
        return str(self.position)

    def remove_from_target(self, final_position):
        return self.position in self.target and final_position not in self.target

    def should_move(self):
        return self.should_move_red() if self.color == RED else self.should_move_green()


    def should_move_red(self):
        if self.position == (0, 12):
            return False

        if self.position == (1, 11) or self.position == (1, 13):
            if self.board.is_ocuppied((0, 12)):
                return False

        if self.position == (2, 10) or self.position == (2, 12) or self.position == (2, 14):
            if self.board.is_ocuppied((0, 12)) and self.board.is_ocuppied((1, 11)) and self.board.is_ocuppied((1, 13)):
                return False

        if self.position == (3, 9) or self.position == (3, 11) or self.position == (3, 13) or self.position == (3, 15):
            if self.board.is_ocuppied((0, 12)) and self.board.is_ocuppied((1, 11)) and self.board.is_ocuppied((1, 13)) and self.board.is_ocuppied((2, 10)) and self.board.is_ocuppied((2, 12)) and self.board.is_ocuppied((2, 14)):
                return False

        return True

    def should_move_green(self):
        if self.position == (16, 12):
            return False

        if self.position == (15, 11) or self.position == (15, 13):
            if self.board.is_ocuppied((16, 12)):
                return False

        if self.position == (14, 10) or self.position == (14, 12) or self.position == (14, 14):
            if self.board.is_ocuppied((16, 12)) and self.board.is_ocuppied((15, 11)) and self.board.is_ocuppied((15, 13)):
                return False

        if self.position == (13, 9) or self.position == (13, 11) or self.position == (13, 13) or self.position == (13, 15):
            if self.board.is_ocuppied((16, 12)) and self.board.is_ocuppied((15, 11)) and self.board.is_ocuppied((15, 13)) and self.board.is_ocuppied((14, 10)) and self.board.is_ocuppied((14, 12)) and self.board.is_ocuppied((14, 14)):
                return False

        return True


    def move(self):
        '''
            Move one space
            Returns possible 1 distance moves
        '''
        possible_moves = set()
        i, j = self.position

        for delta_i, delta_j in MOVES:
            dst_i, dst_j = (i + delta_i, j + delta_j)
            final_position = (dst_i, dst_j)
            if self.board.is_valid(final_position):
                if not self.board.is_ocuppied(final_position) and self.should_move():
                    # Espacio vacío
                    possible_moves.add(final_position)

        return possible_moves

    def jump(self):
        '''
            Make jump
            Returns final position of possible jump moves
            BFS-like algorithm
        '''
        possible_moves = set()
        pending_moves = self.immediate_jump(self.position)
        i, j = self.position
        while pending_moves:
            current_move = pending_moves.pop()
            possible_moves.add(current_move)
            for jump in self.immediate_jump(current_move):
                if jump not in possible_moves:
                    pending_moves.add(jump)
        return possible_moves

    def immediate_jump(self, position):
        possible_moves = set()
        i, j = position

        for delta_i, delta_j in MOVES:
            dst_i, dst_j = (i + delta_i, j + delta_j)
            next_position = (dst_i, dst_j)
            final_position = (dst_i + delta_i, dst_j + delta_j)
            if self.board.is_valid(next_position) and self.board.is_ocuppied(next_position):
                if self.board.is_valid(final_position) and not self.board.is_ocuppied(final_position):
                    # Salto ficha
                    possible_moves.add(final_position)

        return possible_moves

    def possible_moves(self):
        if not self.should_move():
            return set()
        return self.move().union(self.jump())

    def distance_to_zone(self):
        i, j = self.position

        # distance_x = abs(12 - j)
        if self.color == RED:
            distance_y = abs(0 - i)
        else:
            distance_y = abs(16 - i)
        return distance_y ** 2
        # return sqrt((distance_x**2) + (distance_y**2))

    def make_move(self, final_position, calculate_move=False):
        '''
            final_position is an emtpy space and valid move
        '''
        i, j = self.position
        dst_i, dst_j = final_position
        if calculate_move:
            imaginary_board = deepcopy(self.board)
            marbles = imaginary_board.players[self.color].marbles
            for marble in marbles:
                if marble.position == self.position:
                    marble.make_move(final_position)
            return imaginary_board
        else:
            self.board.board[i, j] = EMPTY_SPACE
            self.board.board[dst_i, dst_j] = self.color
            self.previous_position = self.position
            self.position = final_position


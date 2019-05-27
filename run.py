import sys
import datetime
import plotly.graph_objs as go
from plotly.offline import plot
from time import time, sleep
from colorama import init
from src.board import Board
from src.consts import *
from src.util import as_vector, oponent_color
from src.player import Player
from src.neural import NNQFunction
import os

def play_game(dummy=True, weights=None, matches=0, train=True, types=[NEURAL_TYPE, NEURAL_TYPE]):
    board = Board(weights, types)
    red_player = board.players[RED]
    green_player = board.players[GREEN]
    red_won = False
    green_won = False
    moves = 0
    won = False
        

    while moves < MAX_PLAYS:
        moved, _ = red_player.random_move() if dummy else red_player.best_move(learn=False)
        output_board(board, green_player.weights, green_player.moves, matches)
        over, color = is_game_over(board, GREEN)
        if not moved or over:
            break
        moved, match_errors = green_player.best_move(learn=train)
        output_board(board, green_player.weights, green_player.moves, matches)
        over, color = is_game_over(board, GREEN)
        if not moved or over:
            break

        moves = green_player.moves
    if not moves < MAX_PLAYS:
        return TIE, green_player.weights, [0.001]
    elif color == GREEN:
        return WIN, green_player.weights, [0.001]
    elif color == RED:
        return LOSS, green_player.weights, [0.001]

def is_game_over(board, color):
    vector = as_vector(board, color)
    in_target = vector[0]
    oponent_in_target = vector[1]
    if in_target == 10:
        return True, color
    elif oponent_in_target == 10:
        return True, oponent_color(color)
    else:
        return False, None

def training():
    start_time = time()
    matches = 0
    weights = None
    random_won_matches = 0
    random_tied_matches = 0
    random_lost_matches = 0
    match_errors = []
    random_training_errors = []
    last_model = None

    if os.path.exists("model.h5"):
        os.remove("model.h5")
    
    while matches < MAX_MATCHES:
        result, weights, match_errors = play_game(weights=[weights, None], matches=matches, train=False)
        weights.adjust()
        matches += 1
        average_match_error = sum(match_errors) / max(len(match_errors), 1)
        random_training_errors.append(average_match_error)

        weights.save(MODEL_PATH)
        if (matches + 1) % 5 == 0:
            last_model = NNQFunction(RED, MODEL_PATH)

        if result == WIN:
            random_won_matches += 1
        elif result == TIE:
            random_tied_matches += 1
        else:
            random_lost_matches += 1

    # matches = 0
    # won_matches = 0
    # tied_matches = 0
    # lost_matches = 0
    # first_weights = weights
    # weights = None
    # match_errors = []
    # last_training_errors = []

    # while matches < MAX_MATCHES:
    #     result, weights, match_errors = play_game(dummy=False, weights=[weights, first_weights], matches=matches)
    #     matches += 1
    #     average_match_error = sum(match_errors) / len(match_errors)
    #     last_training_errors.append(average_match_error)

    #     if result == WIN:
    #         won_matches += 1
    #     elif result == TIE:
    #         tied_matches += 1
    #     else:
    #         lost_matches += 1

    elapsed_time = time() - start_time
    # player = Player(None, None, None, None)

    print('####### RANDOM #######')
    print(f'WON: {random_won_matches}/{matches}')
    print(f'TIED: {random_tied_matches}/{matches}')
    print(f'LOST: {random_lost_matches}/{matches}')
    print('######################')
    # print('######### LAST #########')
    # print(f'FINAL WEIGHTS: {weights}')
    # print(f'WON: {won_matches}/{matches}')
    # print(f'TIED: {tied_matches}/{matches}')
    # print(f'LOST: {lost_matches}/{matches}')
    # print('########################')
    print(f'TOTAL TIME: {datetime.timedelta(seconds=elapsed_time)}')
    print(f'MAX_PLAYS: {MAX_PLAYS}')
    # print(f'RATE: {player.rate}')
    # print(f'INITIAL WEIGHTS: {player.initial_weights}')

    # make_plot(random_training_errors, last_training_errors)

def play():
    random_weights = [1.0007527530526446, 5.002700604838168, -4.999940205388994, -4.939582748619246, 0.5908794495012145]
    last_weights = [1.006242044202283, 5.0233875752443575, -4.97680697865877, -4.512247513789265, 0.9638332492351976]
    neural_weights = NNQFunction(GREEN, MODEL_PATH)
    
    start_time = time()
    random_matches = 0
    
    neural_random_won_matches = 0
    neural_random_tied_matches = 0
    neural_random_lost_matches = 0

    while random_matches < 100:
        result, _, _ = play_game(weights=[neural_weights, random_weights], matches=random_matches, train=False, types=[NEURAL_TYPE, CLASSIC_TYPE])
        random_matches += 1

        if result == WIN:
            neural_random_won_matches += 1
        elif result == TIE:
            neural_random_tied_matches += 1
        else:
            neural_random_lost_matches += 1

    random_elapsed_time = time() - start_time


    start_time = time()
    matches = 0
    
    neural_won_matches = 0
    neural_tied_matches = 0
    neural_lost_matches = 0

    while matches < 100:
        result, _, _ = play_game(weights=[neural_weights, last_weights], matches=matches, train=False, types=[NEURAL_TYPE, CLASSIC_TYPE])
        matches += 1

        if result == WIN:
            neural_won_matches += 1
        elif result == TIE:
            neural_tied_matches += 1
        else:
            neural_lost_matches += 1

    elapsed_time = time() - start_time


    print('####### RESULTS Neural vs Random #######')
    print(f'WON: {neural_random_won_matches}/{random_matches}')
    print(f'TIED: {neural_random_tied_matches}/{random_matches}')
    print(f'LOST: {neural_random_lost_matches}/{random_matches}')
    print('######################')
    print(f'TOTAL TIME: {datetime.timedelta(seconds=elapsed_time)}')
    print(f'MAX_PLAYS: {MAX_PLAYS}')
    print('####### RESULTS Neural vs Not Random #######')
    print(f'WON: {neural_won_matches}/{matches}')
    print(f'TIED: {neural_tied_matches}/{matches}')
    print(f'LOST: {neural_lost_matches}/{matches}')
    print('######################')
    print(f'TOTAL TIME: {datetime.timedelta(seconds=elapsed_time)}')
    print(f'MAX_PLAYS: {MAX_PLAYS}')

def output_board(board, weights, moves, matches):
    # sleep(1)
    vector = as_vector(board, GREEN)
    print("\033c")
    print(board)
    print(board.players[GREEN].learning_function(as_vector(board, GREEN)))
    print(board.players[GREEN].rate)
    print(board.players[GREEN].initial_weights)
    print(vector)
    print(weights, f'Move #{moves}')
    print(matches)

def make_plot(random_errors, last_errors):
    x = [x for x in range(0, MAX_MATCHES + 1)]

    trace0 = go.Scatter(
        x = x,
        y = random_errors,
        mode = 'lines',
        name = 'lines'
    )

    trace1 = go.Scatter(
        x = x,
        y = last_errors,
        mode = 'lines',
        name = 'lines'
    )

    data = [trace0]
    data1 = [trace1]

    plot(data)
    plot(data1)

if __name__ == "__main__":
    init(autoreset=True)
    training()
    play()

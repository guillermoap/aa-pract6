from math import sqrt
from src.consts import GREEN, RED

def initial_positions(color):
    positions = []
    if color == GREEN:
        positions.append((0,12))
        for w in range(11, 14, 2):
            positions.append((1, w))
        for w in range(10, 15, 2):
            positions.append((2, w))
        for w in range(9, 16, 2):
            positions.append((3, w))
    elif color == RED:
        for w in range(9, 16, 2):
            positions.append((13, w))
        for w in range(10, 15, 2):
            positions.append((14, w))
        for w in range(11, 14, 2):
            positions.append((15, w))
        positions.append((16,12))
    return set(positions)

def oponent_color(color):
    return (color + 2) % 6 + 1

def as_vector(board, color):
    me = board.players[color]
    oponent = board.players[oponent_color(color)]
    my_positions = { marble.position for marble in me.marbles }
    opononet_positions = { marble.position for marble in oponent.marbles }
    #x1 = len(in_target)
    in_target = len(me.target & my_positions)
    #x2 = len(oponent_in_target)
    oponent_in_target = len(oponent.target & opononet_positions)
    # #x3 = total distance to zone
    my_total_distance_to_zone = 0
    for marble in me.marbles:
        my_total_distance_to_zone += marble.distance_to_zone()
    #x4 = total distance to zone
    op_total_distance_to_zone = 0
    for marble in oponent.marbles:
        op_total_distance_to_zone += marble.distance_to_zone()
    
    return (in_target, oponent_in_target, my_total_distance_to_zone ** 2, op_total_distance_to_zone ** 2)

zone_map = {
    GREEN: initial_positions(GREEN),
    RED: initial_positions(RED),
}


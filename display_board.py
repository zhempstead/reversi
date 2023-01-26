import itertools

import numpy as np

from constants import BLACK, WHITE, EMPTY, PLAYER

def board2str(board, other_board=None, key2move=None, player=None):
    '''
    board: current board state
    other_board: a different board state. Diffs between the two will be highlighted.
    key2move: Dict of char -> (int, int). Display key mapping for specific moves
    '''
    key2move = key2move or {}

    b = board.astype(object)
    b[b == BLACK] = '\033[30m●'
    b[b == WHITE] = '\033[37m●'
    b[b == EMPTY] = '\033[30m□'

    if other_board is not None:
        b = np.where(board != other_board, '\033[44m' + b + '\033[42m', b)
    for key, move in key2move.items():
        color = 30 if player == BLACK else 37
        b[move] = f"\033[{color}m{key}"

    return '\n'.join(['\033[42m' + ' '.join(row) + '\033[40m' for row in b]) + '\033[37m'

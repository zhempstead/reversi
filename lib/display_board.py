import itertools

import numpy as np

from .constants import BLACK, WHITE, EMPTY, PLAYER
from .notation import A

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

def board2ascii(board, black_char, white_char, empty_char='.'):
    '''
    board: current board state
    black_char: character to use for black piece
    white_char: character to use for white piece
    empty_char: character to use for empty square
    '''
    b = board.astype(object)
    b[b == BLACK] = black_char
    b[b == WHITE] = white_char
    b[b == EMPTY] = empty_char

    rows = [[' '] + [chr(A + c) for c in range(board.shape[0])]]
    rows += [[str(idx+1)] + list(row) for idx, row in enumerate(b)]
    
    return '\n'.join([' '.join(row) for row in rows])
        

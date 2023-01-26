import itertools

import numpy as np

BLACK = 1
WHITE = -1
EMPTY = 0

class ReversiBoard(object):
    def __init__(self, dim):
        if dim % 2 != 0:
            raise ValueError("Dimensions must be even")
        elif dim < 4:
            raise ValueError("Dimensions must be >= 4")
        self.dim = dim
        self.num_squares = dim*dim
        self.board = np.zeros((dim, dim), dtype=int)
        
        self.board[dim//2 - 1, dim//2 - 1] = WHITE
        self.board[dim//2 - 1, dim//2] = BLACK
        self.board[dim//2, dim//2 - 1] = BLACK
        self.board[dim//2, dim//2] = WHITE

    def legal_moves(self, player):
        if player not in [BLACK, WHITE]:
            raise ValueError("Player must be BLACK or WHITE")
        opponent = -player

        found = set()

        # TODO: optimize for full board
        for i, j in zip(*np.where(self.board == player)):
            for crawl in crawls_from(i, j, self.dim):
                for idx, (i2, j2) in enumerate(crawl):
                    if self.board[i2, j2] == EMPTY:
                        if idx != 0:
                            found.add((i2, j2))
                        break
                    if self.board[i2, j2] == player:
                        break

        return found

    def place_piece(self, row, col, player):
        if self.board[row, col] != EMPTY:
            raise ValueError("Square must be empty")
        self.board[row, col] = player

        opponent = -player
        for crawl in crawls_from(row, col, self.dim):
            flip_i = []
            flip_j = []
            for i, j in crawl:
                if self.board[i, j] == player:
                    self.board[flip_i, flip_j] = player
                    break
                elif self.board[i, j] == EMPTY:
                    break
                flip_i.append(i)
                flip_j.append(j)

    def get_score(self):
        return {p: len(np.where(self.board == p)[0]) for p in [BLACK, WHITE]}

    def get_board(self):
        return self.board.copy()

    def __repr__(self):
        return str(self.board)

def crawls_from(i, j, dim):
    icrawls = [[i]*dim, range(i+1, dim), range(i-1, -1, -1)]
    jcrawls = [[j]*dim, range(j+1, dim), range(j-1, -1, -1)]
    crawl_pairs = list(itertools.product(icrawls, jcrawls))[1:]
    return [zip(*p) for p in crawl_pairs]

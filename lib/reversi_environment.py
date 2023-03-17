import itertools

import numpy as np

from .constants import BLACK, WHITE, EMPTY, PLAYER
from .display_board import board2str

class ReversiEnvironment(object):
    def __init__(self, dim, board=None, curr_player=BLACK, last_moved=True):
        self.dim = dim
        self.board = board
        if self.board is None:
            self.init_board()

        self.curr_player = curr_player
        self.last_moved = last_moved

    def init_board(self):    
        if self.dim % 2 != 0:
            raise ValueError("Dimensions must be even")
        elif self.dim < 4:
            raise ValueError("Dimensions must be >= 4")
        self.board = np.zeros((self.dim, self.dim), dtype=int)
        
        self.board[self.dim//2 - 1, self.dim//2 - 1] = WHITE
        self.board[self.dim//2 - 1, self.dim//2] = BLACK
        self.board[self.dim//2, self.dim//2 - 1] = BLACK
        self.board[self.dim//2, self.dim//2] = WHITE

    def legal_actions(self):
        opponent = -self.curr_player

        found = set()

        # TODO: optimize for full board
        for i, j in zip(*np.where(self.board == self.curr_player)):
            for crawl in crawls_from(i, j, self.dim):
                for idx, (i2, j2) in enumerate(crawl):
                    if self.board[i2, j2] == EMPTY:
                        if idx != 0:
                            found.add((i2, j2))
                        break
                    if self.board[i2, j2] == self.curr_player:
                        break

        return found

    def act(self, action):
        '''
        Returns
        - New environment
        - Reward
        - If game is over
        '''
        new_board = self.board.copy()
        opponent = -self.curr_player
        new_last_moved = True
        reward = 0
        game_over = False

        if action is None:
            new_last_moved = False
            if not self.last_moved:
                game_over = True
                scores = self.get_score()
                if scores[BLACK] > scores[WHITE]:
                    reward = 1
                elif scores[WHITE] > scores[BLACK]:
                    reward = -1
        else:
            row, col = action
            #if self.board[row, col] != EMPTY:
            #    raise ValueError("Square must be empty")

            new_board[row, col] = self.curr_player

            for crawl in crawls_from(row, col, self.dim):
                flip_i = []
                flip_j = []
                for i, j in crawl:
                    if new_board[i, j] == self.curr_player:
                        new_board[flip_i, flip_j] = self.curr_player
                        break
                    elif new_board[i, j] == EMPTY:
                        break
                    flip_i.append(i)
                    flip_j.append(j)

        new_env = ReversiEnvironment(self.dim, new_board, opponent, new_last_moved)
        return (new_env, reward, game_over)

    def get_score(self):
        return {p: len(np.where(self.board == p)[0]) for p in [BLACK, WHITE]}

    def __repr__(self):
        board_str = board2str(self.board)
        text_str = f"{PLAYER[self.curr_player]}'s turn"
        return f"{text_str}\n{board_str}"

def crawls_from(i, j, dim):
    icrawls = [[i]*dim, range(i+1, dim), range(i-1, -1, -1)]
    jcrawls = [[j]*dim, range(j+1, dim), range(j-1, -1, -1)]
    crawl_pairs = list(itertools.product(icrawls, jcrawls))[1:]
    return [zip(*p) for p in crawl_pairs]

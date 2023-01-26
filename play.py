import numpy as np

from board import ReversiBoard, BLACK, WHITE, EMPTY

KEYS = 'qwertyuiopasdfghjklzxcvbnm'

PLAYER = {BLACK: "Erica", WHITE: "Zach"}

class Reversi(object):
    def __init__(self, dim):
        self.board = ReversiBoard(dim)
        self.player = BLACK

    def get_display_board(self):
        b = self.board.get_board().astype(object)
        b[np.where(b == 0)] = '.'
        b[np.where(b == BLACK)] = 'E'
        b[np.where(b == WHITE)] = 'Z'
        return b

    def play(self):
        prev_could_play = True
        while(True):
            self.print_score()
            could_play = self.take_turn(self.player)
            if not could_play and not prev_could_play:
                scores = self.board.get_score()
                if scores[BLACK] > scores[WHITE]:
                    print(f"{PLAYER[BLACK]} wins!")
                elif scores[BLACK] < scores[WHITE]:
                    print(f"{PLAYER[WHITE]} wins!")
                else:
                    print(f"Tie Game!")
            self.player = -self.player
            prev_could_play = could_play


    def print_score(self):
        scores = self.board.get_score()
        for player, score in scores.items():
            print(f"{PLAYER[player]}: {score}")
            

    def take_turn(self, player):
        display_board = self.get_display_board()
        legal_moves = self.board.legal_moves(player)
        key2move = dict(zip(KEYS, legal_moves))
        for key, move in key2move.items():
            display_board[move] = key

        for row in display_board:
            print(' '.join(row))
        print(f"{PLAYER[player]}'s turn")
        if not legal_moves:
            print("No legal moves!")
            return False

        selection = None
        while selection not in key2move.keys():
            selection = input("Move selection: ")
        self.board.place_piece(*key2move[selection], player)
        return True
            

if __name__ == '__main__':
    r = Reversi(8)
    r.play()

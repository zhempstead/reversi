import json
import sys
import time

import numpy as np

from constants import BLACK, WHITE, PLAYER
from display_board import board2str
from reversi_environment import ReversiEnvironment

KEYS = 'qwertyuiopasdfghjklzxcvbnm'

class Replay(object):
    def __init__(self, record_file):

        print(f"Reading game from '{record_file}'")
        with open(record_file) as record_fp:
            record = json.load(record_fp)
        self.dim = record['dim']
        self.actions = record['actions']

        self.env = ReversiEnvironment(dim=self.dim)
        self.prev_env = self.env

    def play(self):
        turn = 0
        while(True):
            print(f"Turn {turn}")
            self.print_score()
            print(board2str(self.env.board, self.prev_env.board))
            input("Continue: ")
            action = self.actions[turn][0]
            self.prev_env = self.env
            self.env, reward, game_over = self.env.act(action)
            if game_over:
                self.game_over(reward)
                break
            
            turn += 1

    def game_over(self, reward):
        self.print_score()
        if reward == 0:
            print("Tie game!")
        else:
            print(f"{PLAYER[reward]} wins!")


    def print_score(self):
        scores = self.env.get_score()
        for player, score in scores.items():
            print(f"{PLAYER[player]}: {score}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        replay = sys.argv[1]
    else:
        replay = 'replays/human.json'
    r = Replay(replay)
    r.play()

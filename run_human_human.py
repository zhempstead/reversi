import json
import time

import numpy as np

from constants import BLACK, WHITE, PLAYER
from display_board import board2str
from reversi_environment import ReversiEnvironment

KEYS = 'qwertyuiopasdfghjklzxcvbnm'

class HumanHuman(object):
    def __init__(self, dim, record_file=None):
        self.env = ReversiEnvironment(dim=dim)
        self.prev_env = self.env

        self.record = {'dim': dim, 'actions': []}
        self.record_file = record_file

    def play(self):
        while(True):
            self.print_score()
            action = self.choose_action()
            self.prev_env = self.env
            self.env, reward, game_over = self.env.act(action)
            self.record['actions'].append((action, reward))
            if game_over:
                self.game_over(reward)
                break

    def choose_action(self):
        player = self.env.curr_player
        legal_actions = sorted(self.env.legal_actions())
        if not legal_actions:
            print(board2str(self.env.board))
            print(f"No legal actions for {PLAYER[player]}!")
            self.pause()
            return None
        if len(legal_actions) > len(KEYS):
            raise ValueError("Too many legal actions to represent")
        key2action = dict(zip(KEYS, legal_actions))
        print(board2str(self.env.board, self.prev_env.board, key2action, player))
        print(f"{PLAYER[player]}'s turn")

        selection = None
        while selection not in key2action.keys():
            selection = input("Move selection: ")
        return key2action[selection]

    def game_over(self, reward):
        self.print_score()
        if reward == 0:
            print("Tie game!")
        else:
            print(f"{PLAYER[reward]} wins!")
        if self.record_file is not None:
            with open(self.record_file, 'w') as record_json:
                record_json.write(json.dumps(self.record))

    
    def print_score(self):
        scores = self.env.get_score()
        for player, score in scores.items():
            print(f"{PLAYER[player]}: {score}")
            
    def pause(self):
        time.sleep(0.5)

if __name__ == '__main__':
    r = HumanHuman(6)
    r.play()

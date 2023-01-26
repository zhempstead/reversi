import json

import numpy as np

from agents import DualAgent, GreedyAgent, HumanAgent, RandomAgent
from constants import PLAYER
from reversi_environment import ReversiEnvironment

class ReversiGame(object):
    def __init__(self, dim, agent, record_file=None, headless=False):
        self.env = ReversiEnvironment(dim=dim)
        self.prev_env = self.env
        self.agent = agent
        self.turn = 0

        self.headless = headless

        self.record = {'dim': dim, 'actions': []}
        self.record_file = record_file
        if self.record_file:
            print(f"Recording game at '{self.record_file}'")


    def play(self):
        while(True):
            print(f"Turn {self.turn}")
            if not self.headless:
                self.print_score()
            action = self.agent.choose_action(self.env, self.prev_env)
            self.prev_env = self.env
            self.env, reward, game_over = self.env.act(action)
            self.record['actions'].append((action, reward))
            if game_over:
                self.game_over(reward)
                break
            self.turn += 1

    def game_over(self, reward):
        if not self.headless:
            self.print_score()
            if reward == 0:
                print("Tie game!")
            else:
                print(f"{PLAYER[reward]} wins!")
        if self.record_file is not None:
            with open(self.record_file, 'w') as record_fp:
                json.dump(self.record, record_fp, cls=NpEncoder)

    
    def print_score(self):
        scores = self.env.get_score()
        for player, score in scores.items():
            print(f"{PLAYER[player]}: {score}")
            
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    random = RandomAgent()
    greedy = GreedyAgent()
    human = HumanAgent()
    rg = ReversiGame(8, DualAgent(greedy, human), 'replays/dual.json', headless=False)
    #rg = ReversiGame(8, HumanAgent(), 'replays/human.json', headless=False)
    rg.play()

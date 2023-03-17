import json

import numpy as np
import pandas as pd

from agents import DualAgent, GPTAgent, HumanAgent, RandomAgent, ScoreGreedyAgent, ScoreMinimaxAgent
from constants import PLAYER, BLACK, WHITE
from replay import Replay
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
                return reward
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


def gpt_game(dims, other_agent, gpt_player, shots, replay, record_file=None):
    gpt = GPTAgent(model="text-davinci-003", learning_shots=shots, replay=replay)
    if gpt_player == BLACK:
        dual = DualAgent(gpt, other_agent)
    else:
        dual = DualAgent(other_agent, gpt)
    rg = ReversiGame(dims, dual, headless=True, record_file=record_file)
    try:
        result = rg.play()
    except ValueError as e:
        print(e)
        result = 2
    return result

if __name__ == '__main__':
    replay = Replay("replays/authoritative_8.json")

    '''
    random = RandomAgent()
    greedy = ScoreGreedyAgent()
    human = HumanAgent()
    gpt = GPTAgent(learning_shots=2, replay=replay)
    minimax_1 = ScoreMinimaxAgent(1)
    minimax_2 = ScoreMinimaxAgent(-1)
    #for mmx in range(100):
    #    print(f"{mmx} forward look")
    #    minimax = ScoreMinimaxAgent(mmx)
    #    rg = ReversiGame(6, minimax, f'replays/authoritative_{mmx}.json', headless=False)
    #    rg.play()
    rg = ReversiGame(6, DualAgent(random, gpt), 'replays/rand_v_gpt.json', headless=False)
    #rg = ReversiGame(8, HumanAgent(), 'replays/human.json', headless=False)
    rg.play()
    '''

    results = {'opponent': [], 'shots': [], 'gpt_player': [], 'result': []}
    random = RandomAgent()
    greedy = ScoreGreedyAgent()
    minimax3 = ScoreMinimaxAgent(3)
    for i in range(10):
        for shots in range(3):
            for gpt_player in [BLACK, WHITE]:
                result = gpt_game(6, minimax3, gpt_player, shots, replay, f'replays/gpt_{i}_{shots}_{gpt_player}.json')
                results['opponent'].append('minimax3')
                results['shots'].append(shots)
                results['gpt_player'].append(gpt_player)
                results['result'].append(result)
    pd.DataFrame(results).to_csv('gpt_results.csv', index=False)

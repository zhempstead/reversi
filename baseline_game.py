import json
import sys

import numpy as np
import pandas as pd

from lib.agents import DualAgent, GPTAgent, RandomAgent, ScoreGreedyAgent, ScoreMinimaxAgent
from lib.constants import PLAYER, BLACK, WHITE
from lib.replay import Replay
from lib.reversi_game import ReversiGame


results = {'player_B': [], 'player_W': [], 'victor': []}
random = RandomAgent()
greedy = ScoreGreedyAgent()
minimax3 = ScoreMinimaxAgent(3)
agents = {'random': random, 'greedy': greedy, 'minimax3': minimax3}

def summarize_results(df):
    df['player1'] = df[['player_B', 'player_W']].min(axis=1)
    df['player2'] = df[['player_B', 'player_W']].max(axis=1)
    df = df.drop(columns=['player_B', 'player_W'])
    df['count'] = 1
    df = df.groupby(['player1', 'player2', 'victor']).count()
    print(df)
    
if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1])
    summarize_results(df)
    sys.exit()

for i in range(100):
    for black_pname, black_agent in agents.items():
        for white_pname, white_agent in agents.items():
            if black_pname == white_pname:
                continue
            dual = DualAgent(black_agent, white_agent)
            rg = ReversiGame(6, dual, headless=True, record_file = f'replays/{black_pname}_{white_pname}_{i}.json')
            result = rg.play()

            results['player_B'].append(black_pname)
            results['player_W'].append(white_pname)
            if result == BLACK:
                results['victor'].append(black_pname)
            elif result == WHITE:
                results['victor'].append(white_pname)
            else:
                results['victor'].append('Tie')
df = pd.DataFrame(results)
df.to_csv('baseline_results.csv', index=False)


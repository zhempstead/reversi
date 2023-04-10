import json

import numpy as np
import pandas as pd

from lib.agents import DualAgent, GreedyGPTVisualAgent, GPTAgent, RandomAgent, ScoreGreedyAgent, ScoreMinimaxAgent
from lib.constants import PLAYER, BLACK, WHITE
from lib.replay import Replay
from lib.reversi_game import ReversiGame

def gpt_game(dims, other_agent, gpt_player, shots, replay, record_file=None):
    gpt = GreedyGPTVisualAgent(model="gpt-3.5-turbo", learning_shots=shots, replay=replay)
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

# Replace this with any replay suitable for generating learning examples for GPT
replay = Replay("replays/authoritative_8.json")

results = {'opponent': [], 'shots': [], 'gpt_player': [], 'victor': []}
random = RandomAgent()
greedy = ScoreGreedyAgent()
minimax3 = ScoreMinimaxAgent(3)
try:
    for i in range(10):
        for shots in (2, 1, 0):
            for gpt_player in [BLACK, WHITE]:
                #for opponent, opp_agent in [('random', random), ('greedy', greedy), ('minimax3', minimax3)]:
                for opponent, opp_agent in [('random', random), ('greedy', greedy)]:
                    result = gpt_game(6, opp_agent, gpt_player, shots, replay, f'replays/gpt_{i}_{opponent}_{shots}_{gpt_player}.json')
                    results['opponent'].append(opponent)
                    results['shots'].append(shots)
                    results['gpt_player'].append(gpt_player)
                    results['victor'].append(result)
finally:
    df = pd.DataFrame(results)
    df['result'] = 'Tie'
    df.loc[df['gpt_player'] == df['victor'], 'result'] = 'Win'
    df.loc[df['gpt_player'] == -df['victor'], 'result'] = 'Loss'
    df.loc[df['victor'] == 2, 'result'] = 'Invalid'
    df['gpt_player'].apply(lambda x: PLAYER[x][0])
    df = df.drop(columns=['victor'])
    df.to_csv('gpt_greedy_results.csv', index=False)

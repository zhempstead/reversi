import json

import numpy as np
import pandas as pd

from lib.agents import DualAgent, MinimaxGPTAgent, RandomAgent, ScoreGreedyAgent, ScoreMinimaxAgent
from lib.constants import PLAYER, BLACK, WHITE
from lib.replay import Replay
from lib.reversi_game import ReversiGame

def gpt_game(dims, other_agent, gpt_player, search_depth, lookahead, record_file=None):
    gpt = MinimaxGPTAgent(model="gpt-3.5-turbo", search_depth=search_depth, lookahead=lookahead)
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

results = {'opponent': [], 'search_depth': [], 'lookahead': [], 'gpt_player': [], 'victor': []}
random = RandomAgent()
greedy = ScoreGreedyAgent()
minimax1 = ScoreMinimaxAgent(1)
minimax2 = ScoreMinimaxAgent(2)
minimax3 = ScoreMinimaxAgent(3)
minimax4 = ScoreMinimaxAgent(4)
try:
    for i in range(5):
        for gpt_player in [BLACK, WHITE]:
            for opponent, opp_agent in [('minimax1', minimax1)]:
            #for opponent, opp_agent in [('minimax1', minimax1), ('minimax2', minimax2), ('minimax3', minimax3), ('minimax4', minimax4)]:
                for search_depth in [1,2,3,4]:
                    for lookahead in [1,2,3,4]:
                        result = gpt_game(6, opp_agent, gpt_player, search_depth, lookahead, f'replays/gpt_minimax_{i}_{opponent}_{search_depth}_{lookahead}_{gpt_player}.json')
                        results['opponent'].append(opponent)
                        results['search_depth'].append(search_depth)
                        results['lookahead'].append(lookahead)
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
    df.to_csv('gpt_minimax_results.csv', index=False)

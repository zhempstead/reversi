import pandas as pd

from lib.agents import DualAgent, ScoreGreedyAgent, ScoreMinimaxAgent
from lib.gpt_query import legal_query
from lib.notation import extract_all_notation
from lib.replay import Replay
from lib.reversi_game import ReversiGame


# Replace this with any replay suitable for generating learning examples for GPT
learning_replay = Replay("replays/authoritative_8.json")

results = {'shots': [], 'turn': [], 'legal_count': [], 'true_positive': [], 'false_positive': [], 'false_negative': [], 'game': [], 'visualize': []}
greedy = ScoreGreedyAgent()
minimax3 = ScoreMinimaxAgent(3)
try:
    for i in range(1):
        replay_file = f'replays/legal_ref_{i}.json'
        dual = DualAgent(greedy, minimax3)
        #rg = ReversiGame(6, dual, headless=True, record_file = f'replays/legal_ref_{i}.json')
        #rg.play()

        replay = Replay(replay_file)
        for turn in range(len(replay.actions)):
            env = replay.state_before_turn(turn)
            for shots in range(3):
                for viz in [True, False]:
                    resp = legal_query("gpt-3.5-turbo", env, shots, learning_replay, visualize=viz)
                    resp_moves = extract_all_notation(resp)
                    legal_moves = env.legal_actions()
                    print(f"Turn {turn}")
                    print(f"  Legal moves: {legal_moves}")
                    print(f"  Response: {resp_moves}")
                    print(f"  Raw response: {resp}")

                    results['game'].append(i)
                    results['visualize'].append(viz)
                    results['shots'].append(shots)
                    results['turn'].append(turn)
                    results['legal_count'].append(len(legal_moves))
                    results['true_positive'].append(len([m for m in resp_moves if m in legal_moves]))
                    results['false_positive'].append(len([m for m in resp_moves if m not in legal_moves]))
                    results['false_negative'].append(results['legal_count'][-1] - results['true_positive'][-1])
finally:
    df = pd.DataFrame(results)
    df.to_csv('gpt_legal_results_raw.csv', index=False)
    dfg = df.groupby(['visualize', 'shots']).sum()
    dfg['fscore'] = 2*dfg['true_positive'] / (2*dfg['true_positive'] + dfg['false_positive'] + dfg['false_negative'])
    dfg.to_csv('gpt_legal_results.csv')

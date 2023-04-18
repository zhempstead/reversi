import pandas as pd
import random

from lib.agents import DualAgent, ScoreGreedyAgent, ScoreMinimaxAgent
from lib.gpt_query import accurate_move_query, show_hypothetical_move
from lib.notation import extract_all_notation
from lib.replay import Replay
from lib.reversi_game import ReversiGame


# Replace this with any replay suitable for generating learning examples for GPT
learning_replay = Replay("replays/authoritative_8.json")
replays = [Replay(f"replays/authoritative_{i}.json") for i in range(8)]

results = {'replay': [], 'shots': [], 'turn': [], 'accurate': [], 'diffs': []}
try:
    for replay_idx in range(8):
        replay = replays[replay_idx]
        for shots in range(1, 5):
            for turn in range(len(replay.actions)):
                if replay.get_action(turn) is None:
                    continue
                while True:
                    ex_replay_idx = random.randrange(8)
                    if ex_replay_idx != replay_idx:
                        break
                ex_replay = replays[ex_replay_idx]
                resp = accurate_move_query("gpt-3.5-turbo", replay, turn, shots=shots, example_replay=ex_replay).rstrip()
                truth = show_hypothetical_move(replay.state_before_turn(turn+1), replay.state_before_turn(turn).curr_player)
                print(truth)
                results['accurate'].append(resp == truth)
                results['diffs'].append(sum(1 for a, b in zip(truth, resp) if a != b))
                results['replay'].append(replay_idx)
                results['turn'].append(turn)
                results['shots'].append(shots)
finally:
    df = pd.DataFrame(results)
    df.to_csv('gpt_accurate_move.csv', index=False)

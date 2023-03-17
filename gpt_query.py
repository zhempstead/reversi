from dotenv import load_dotenv
import numpy as np
import os
import openai

from constants import PLAYER

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def preamble(env):
    return f"""You are an expert at Reversi (also known as Othello). We are playing on a board that is {env.dim}-by-{env.dim}. All piece positions are 1-indexed, so [1, 1] means the upper-left corner of the board."""

def move_prompt(env, legal_actions):
    return f"""Here are the positions of your pieces: {piece_list(env, env.curr_player)}
Here are the positions of your opponent's pieces: {piece_list(env, env.curr_player*-1)}

It is your turn. Here are your legal moves: {poslist2str(legal_actions)}

As an expert Reversi player, your best move is ["""

def saved_prompt(replay, turn):
    env = replay.state_before_turn(turn)
    legal_actions = env.legal_actions()
    action = replay.get_action(turn)
    return f"{move_prompt(env, legal_actions)}{action[0]}, {action[1]}]."

def move_query(model, env, legal_moves, shots=0, replay=None):
    prompt = [preamble(env)]
    for shot in range(shots):
        turn = shot*3 + 2
        prompt.append(saved_prompt(replay, turn))
        prompt.append("Okay, now let's imagine a new scenario.")

    prompt.append(move_prompt(env, legal_moves))
    return query(model, "\n\n".join(prompt))

def query(model, prompt):
    response = openai.Completion.create(model=model, prompt=prompt, temperature=0, max_tokens=10)
    return response['choices'][0]['text'].strip()

def piece_list(env, player):
    return poslist2str(zip(*np.where(env.board == player)))

def poslist2str(poslist):
    return ", ".join([f"[{x+1}, {y+1}]" for x, y in poslist])

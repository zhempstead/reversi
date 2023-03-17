from dotenv import load_dotenv
import numpy as np
import os
import openai

from .constants import PLAYER

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def preamble(env):
    return f"""You are an expert at Reversi (also known as Othello). We are playing on a board that is {env.dim}-by-{env.dim}. All piece positions are 1-indexed, so [1, 1] means the upper-left corner of the board."""

def move_prompt(env, legal_actions):
    return f"""Here are the positions of your pieces: {piece_list(env, env.curr_player)}
Here are the positions of your opponent's pieces: {piece_list(env, env.curr_player*-1)}

It is your turn. Here are your legal moves: {poslist2str(legal_actions)}

As an expert Reversi player, what is your best move? Just respond with the move itself."""

def example_conversation(replay, turn):
    env = replay.state_before_turn(turn)
    legal_actions = env.legal_actions()
    action = replay.get_action(turn)
    return (move_prompt(env, legal_actions), f'[{action[0]}, {action[1]}].')

def move_query(model, env, legal_moves, shots=0, replay=None):
    prompt = preamble(env)
    messages = []
    for shot in range(shots):
        turn = shot*3 + 2
        messages += example_conversation(replay, turn)
    messages.append(move_prompt(env, legal_moves))
    return query(model, prompt, messages)

def query(model, prompt, conversation):
    messages = [{"role": "system", "content": prompt}]
    for idx, message in enumerate(conversation):
        if idx % 2 == 0:
            role = "user"
        else:
            role = "assistant"
        messages.append({"role": role, "content": message})
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=25)
    return response.choices[0].message.content.strip()

def piece_list(env, player):
    return poslist2str(zip(*np.where(env.board == player)))

def poslist2str(poslist):
    return ", ".join([f"[{x+1}, {y+1}]" for x, y in poslist])

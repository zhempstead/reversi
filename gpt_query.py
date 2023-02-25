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
    print(PLAYER[env.curr_player])
    return f"""Here are the positions of your pieces: {piece_list(env, env.curr_player)}
Here are the positions of your opponent's pieces: {piece_list(env, env.curr_player*-1)}

It is your turn. Here are your legal moves: {poslist2str(legal_actions)}

As an expert Reversi player, your best move is ["""

def move_query(env, legal_moves):
    prompt = preamble(env) + "\n" + move_prompt(env, legal_moves)
    return query(prompt)

def query(prompt):
    print(prompt)
    response = openai.Completion.create(model="text-curie-001", prompt=prompt, temperature=0)
    print(response['choices'][0]['text'].strip())
    return response['choices'][0]['text'].strip()

def piece_list(env, player):
    return poslist2str(zip(*np.where(env.board == player)))

def poslist2str(poslist):
    return ", ".join([f"[{x+1}, {y+1}]" for x, y in poslist])

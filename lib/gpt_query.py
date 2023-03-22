from dotenv import load_dotenv
import numpy as np
import os
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .constants import PLAYER, BLACK, WHITE
from .display_board import board2ascii
from .notation import coords2notation, max_col

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def preamble(env):
    return f"""You are an expert at Reversi (also known as Othello). We are playing on a board that is {env.dim}-by-{env.dim}. We use standard Othello notation where rows are numbered from top to bottom by 1 to {env.dim}, and columns are indicated from left to right by 'a' through '{max_col(env.dim)}'. So for instance, 'b3' denotes the square in the second column from the left and the third row from the top."""

def board_description(env):
    return f"""Here are the positions of your pieces: {piece_list(env, env.curr_player)}
Here are the positions of your opponent's pieces: {piece_list(env, env.curr_player*-1)}"""

def board_visualization(env):
    black_char = 'X' if env.curr_player == BLACK else 'O'
    white_char = 'X' if env.curr_player == WHITE else 'O'
    return f"""A representation of the board follows. Empty squares are denoted by '.', your pieces are denoted by 'X', and your opponent's by 'O'. Columns and rows are labeled. Here is the board:

{board2ascii(env.board, black_char, white_char, '.')}"""

def move_prompt(env, legal_actions, visualize=False):
    state_prompt = board_visualization(env) if visualize else board_description(env)
    return f"""{state_prompt}
    
It is your turn. Here are your legal moves: {poslist2str(legal_actions)}

As an expert Reversi player, what is your best move? Just respond with the move itself."""

def legal_prompt(env, visualize=False):
    state_prompt = board_visualization(env) if visualize else board_description(env)
    return f"""{state_prompt}

It is your turn. List all of your legal moves (say 'None' if you have no legal moves)."""


def example_move_conversation(replay, turn, visualize=False):
    env = replay.state_before_turn(turn)
    legal_actions = env.legal_actions()
    action = replay.get_action(turn)
    return (move_prompt(env, legal_actions, visualize), f"{coords2notation(action)}.")

def example_legal_conversation(replay, turn, visualize=False):
    env = replay.state_before_turn(turn)
    legal_actions = env.legal_actions()
    return (legal_prompt(env, visualize), poslist2str(legal_actions))

def move_query(model, env, legal_moves, shots=0, replay=None, visualize=False):
    prompt = preamble(env)
    messages = []
    for shot in range(shots):
        turn = shot*3 + 2
        messages += example_move_conversation(replay, turn, visualize)
    messages.append(move_prompt(env, legal_moves, visualize))
    return query(model, prompt, messages)

def legal_query(model, env, shots=0, replay=None, visualize=False):
    prompt = preamble(env)
    messages = []
    for shot in range(shots):
        turn = shot*3 + 2
        messages += example_legal_conversation(replay, turn, visualize)
    messages.append(legal_prompt(env, visualize))
    print(legal_prompt(env, visualize))
    return query(model, prompt, messages, max_tokens=100)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query(model, prompt, conversation, max_tokens=25):
    messages = [{"role": "system", "content": prompt}]
    for idx, message in enumerate(conversation):
        if idx % 2 == 0:
            role = "user"
        else:
            role = "assistant"
        messages.append({"role": role, "content": message})
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=max_tokens)
    return response.choices[0].message.content.strip()

def piece_list(env, player):
    return poslist2str(zip(*np.where(env.board == player)))

def poslist2str(poslist):
    return ", ".join([coords2notation(coords) for coords in poslist])

from dotenv import load_dotenv
import numpy as np
import os
import openai
import random
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

def show_hypothetical_move(env, gpt_player):
    black_char = 'X' if gpt_player == BLACK else 'O'
    white_char = 'X' if gpt_player == WHITE else 'O'
    return board2ascii(env.board, black_char, white_char, '.')

def move_prompt(env, legal_actions, visualize=False):
    state_prompt = board_visualization(env) if visualize else board_description(env)
    return f"""{state_prompt}
 
It is your turn. Here are your legal moves: {poslist2str(legal_actions)}.

Explain which of your opponent's pieces will be flipped by each legal move."""

def greedy_prompt(env, legal_actions, visualize=False):
    state_prompt = board_visualization(env) if visualize else board_description(env)
    return f"""{state_prompt}

It is your turn. Here are your legal moves: {poslist2str(legal_actions)}.

Describe which pieces will be flipped by each of these legal moves. Then, in one sentence, tell me the move that flips the most pieces (if there's a tie, list all of the equally-good moves).
"""

def greedy_visual_prompt(env, legal_actions):
    state_prompt = board_visualization(env)
    return f"""{state_prompt}

It is your turn. Here are your legal moves: {poslist2str(legal_actions)}.

Show the resulting board state for each legal move. Then, in one sentence, tell me the move that flips the most pieces (if there's a tie, list all of the equally-good moves).
"""

def minimax_prompt(env, actions2outcomes):
    legal_actions = list(actions2outcomes.keys())
    prompts = [board_visualization(env)]
    action_prompts = []
    prompts.append(f"It is your turn.\nHere are your legal moves: {poslist2str(legal_actions)}.\nBelow, we predict the likely game state a few turns in the future for each legal choice.")
    for action, (action_env, result) in actions2outcomes.items():
        state = f"""Game state after a few turns if you choose {coords2notation(action)}:
{show_hypothetical_move(action_env, env.curr_player)}"""
        if result == env.curr_player:
            state += "\n(the game would end with you winning)"
        elif result == -env.curr_player:
            state += "\n(the game would end with you losing)"
        prompts.append(state)
    prompts.append(f"Based on the above, which move is best?")
    return "\n\n".join(prompts)
        

def legal_prompt(env, visualize=False):
    state_prompt = board_visualization(env) if visualize else board_description(env)
    return f"""{state_prompt}

It is your turn. List all of your legal moves (say 'None' if you have no legal moves), explaining which pieces will be flipped and why."""

def accurate_move_prompt(env, action):
    state_prompt = board_visualization(env)
    return f"""{state_prompt}

It is your turn. Show me the resulting board state if you were to choose {coords2notation(action)} as your next move."""


def example_move_conversation(replay, turn, visualize=False):
    env = replay.state_before_turn(turn)
    legal_actions = env.legal_actions()
    action = replay.get_action(turn)
    return (move_prompt(env, legal_actions, visualize), f"{coords2notation(action)}.")

def example_legal_conversation(replay, turn, visualize=False):
    env = replay.state_before_turn(turn)
    legal_actions = env.legal_actions()
    return (legal_prompt(env, visualize), poslist2str(legal_actions))

def example_accurate_move_prompt_conversation(replay, turn):
    env = replay.state_before_turn(turn)
    action = replay.get_action(turn)
    result_env = replay.state_before_turn(turn+1)
    return (accurate_move_prompt(env, action), show_hypothetical_move(result_env, env.curr_player))
    

def example_greedy_conversation(replay, turn, visualize=False):
    env = replay.state_before_turn(turn)
    legal_actions = env.legal_actions()
    best_num_flips = 0
    best_actions = []
    action2flips = {}
    action2num_flips = {}
    for action in legal_actions:
        flips = env.check_flips(action)
        action2flips[action] = flips
        num_flips = sum([len(fs) for _, fs in flips])
        action2num_flips[action] = num_flips
        if num_flips > best_num_flips:
            best_num_flips = num_flips
            best_actions = []
        if num_flips == best_num_flips:
            best_actions.append(action)
    resp = []
    for action in legal_actions:
        resp.append(f"For move {coords2notation(action)}:")
        for surround, flipped in action2flips[action]:
            resp.append(f"- My piece at {coords2notation(surround)} means my opponent's pieces at {', '.join([coords2notation(f) for f in flipped])} are outflanked and flipped.")
        resp.append(f"So this move flips {action2num_flips[action]} in total.\n")

    if len(best_actions) == 1:
        resp.append(f"So the best move is {coords2notation(best_actions[0])}.")
    else:
        resp.append(f"So the best moves are {', '.join([coords2notation(a) for a in best_actions])}.")

    return (greedy_prompt(env, legal_actions, visualize), '\n'.join(resp))

def example_greedy_visual_conversation(replay, turn):
    env = replay.state_before_turn(turn)
    gpt_player = env.curr_player
    legal_actions = env.legal_actions()
    best_num_flips = 0
    best_actions = []
    action2flips = {}
    action2num_flips = {}
    for action in legal_actions:
        flips = env.check_flips(action)
        action2flips[action] = flips
        num_flips = sum([len(fs) for _, fs in flips])
        action2num_flips[action] = num_flips
        if num_flips > best_num_flips:
            best_num_flips = num_flips
            best_actions = []
        if num_flips == best_num_flips:
            best_actions.append(action)
    resp = []
    for action in legal_actions:
        resp.append(f"For move {coords2notation(action)}:")
        (hyp_env, _, _) = env.act(action)
        resp.append(show_hypothetical_move(hyp_env, gpt_player))
        resp.append(f"So this move flips {action2num_flips[action]} O pieces in total.\n")

    if len(best_actions) == 1:
        resp.append(f"So the best move is {coords2notation(best_actions[0])}.")
    else:
        resp.append(f"So the best moves are {', '.join([coords2notation(a) for a in best_actions])}.")

    return (greedy_visual_prompt(env, legal_actions), '\n'.join(resp))


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
    return query(model, prompt, messages, max_tokens=300)

def accurate_move_query(model, replay, turn, shots=0, example_replay=None):
    prompt = preamble(replay.state_before_turn(turn))
    messages = []
    ex_turns = []
    for shot in range(shots):
        while(True):
            ex_turn = random.randrange(len(example_replay.actions) - 1)
            if ex_turn in ex_turns:
                continue
            ex_action = example_replay.get_action(ex_turn)
            if ex_action is not None:
                break
        ex_turns.append(ex_turn)
        messages += example_accurate_move_prompt_conversation(example_replay, ex_turn)
    messages.append(accurate_move_prompt(replay.state_before_turn(turn), replay.get_action(turn)))
    return query(model, prompt, messages, max_tokens=150, strip=False)

def greedy_query(model, env, legal_moves, shots=0, replay=None, visualize=False):
    prompt = preamble(env)
    messages = []
    for shot in range(shots):
        turn = shot*3 + 2
        messages += example_greedy_conversation(replay, turn, visualize)
    messages.append(greedy_prompt(env, legal_moves, visualize))
    return query(model, prompt, messages, max_tokens=1000)

def greedy_visual_query(model, env, legal_moves, shots=0, replay=None):
    prompt = preamble(env)
    messages = []
    for shot in range(shots):
        turn = shot*3 + 2
        messages += example_greedy_visual_conversation(replay, turn)
    messages.append(greedy_visual_prompt(env, legal_moves))
    return query(model, prompt, messages, max_tokens=1000)

def minimax_query(model, env, moves2outcomes, shots=0, replay=None):
    if shots > 0:
        raise NotImplementedError("minimax_query not compatible with few-shot learning")
    prompt = preamble(env)
    messages = [minimax_prompt(env, moves2outcomes)]
    return query(model, prompt, messages, max_tokens=60)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query(model, prompt, conversation, max_tokens=25, strip=True):
    messages = [{"role": "system", "content": prompt}]
    for idx, message in enumerate(conversation):
        if idx % 2 == 0:
            role = "user"
            print('\033[32m' + message + '\033[0m')
        else:
            role = "assistant"
            print("-----")
            print('\033[34m' + message + '\033[0m')
            print("-----")
        messages.append({"role": role, "content": message})
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=max_tokens)

    resp = response.choices[0].message.content
    if strip:
        resp = resp.strip()
    print("-----")
    print('\033[36m' + resp + '\033[0m')
    return resp

def piece_list(env, player):
    return poslist2str(zip(*np.where(env.board == player)))

def poslist2str(poslist):
    return ", ".join([coords2notation(coords) for coords in poslist])

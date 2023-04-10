import random
import time

from .alphabeta import alphabeta
from .constants import BLACK, WHITE, PLAYER
from .display_board import board2str
from .notation import extract_notation, extract_all_notation
from . import gpt_query

class ReversiAgent(object):
    def __init__(self):
        pass

    def choose_action(self, env, prev_env):
        legal_actions = env.legal_actions()
        if not legal_actions:
            return self.forced_pass(env)
        return self.policy(legal_actions, env, prev_env)

    def forced_pass(self, env):
        return None

    def policy(self, legal_actions, env, prev_env):
        raise NotImplementedError()


KEYS = 'qwertyuiopasdfghjklzxcvbnm'

class HumanAgent(ReversiAgent):
    def forced_pass(self, env):
        print(board2str(env.board))
        print(f"No legal actions for {PLAYER[env.curr_player]}!")
        self.pause()
        super().forced_pass(env)

    def policy(self, legal_actions, env, prev_env):
        player = env.curr_player
        legal_actions = sorted(legal_actions)
        
        if len(legal_actions) > len(KEYS):
            raise ValueError("Too many legal actions to represent")
        key2action = dict(zip(KEYS, legal_actions))
        print(board2str(env.board, prev_env.board, key2action, player))
        print(f"{PLAYER[player]}'s turn")

        selection = None
        while selection not in key2action.keys():
            selection = input("Move selection: ")
        return key2action[selection]
    
    def pause(self):
        time.sleep(0.5)

class GPTAgent(ReversiAgent):

    def __init__(self, model="gpt-3.5-turbo", learning_shots=0, replay=None, visualize=False):
        self.model = model
        self.learning_shots=learning_shots
        self.replay = replay
        self.visualize = visualize

    def forced_pass(self, env):
        print(f"No legal actions for GPT Agent so no query sent")
        super().forced_pass(env)

    def policy(self, legal_actions, env, prev_env):
        if len(legal_actions) == 1:
            print(f"Single legal action - no query sent")
            return list(legal_actions)[0]
        response = self.query(env, legal_actions)
        move = self.parse_response(response, legal_actions)
        return move

    def query(self, env, legal_actions):
        return gpt_query.move_query(self.model, env, legal_actions, self.learning_shots, self.replay, self.visualize)
    
    def parse_response(self, response, legal_actions):
        match = extract_notation(response)
        if match is None:
            raise ValueError(f"Couldn't parse GPT response '{response}'")
        return match

class GreedyGPTAgent(GPTAgent):
    def query(self, env, legal_actions):
        return gpt_query.greedy_query(self.model, env, legal_actions, self.learning_shots, self.replay, self.visualize)

    def parse_response(self, response, legal_actions):
        last_clause = response.split('\n\n')[-1]
        moves = extract_all_notation(last_clause)
        moves = [m for m in moves if m in legal_actions]
        if moves:
            return random.choice(moves)
        elif "equal" in last_clause or "same" in last_clause:
            return random.choice(list(legal_actions))
        else:
            raise ValueError(f"No legal move in final clause '{last_clause}'")

class GreedyGPTVisualAgent(GreedyGPTAgent):
    def __init__(self, model="gpt-3.5-turbo", learning_shots=0, replay=None):
        return super().__init__(model=model, learning_shots=learning_shots, replay=replay, visualize=True)

    def query(self, env, legal_actions):
        return gpt_query.greedy_visual_query(self.model, env, legal_actions, self.learning_shots, self.replay)

class MinimaxGPTAgent(GPTAgent):
    def __init__(self, model="gpt-3.5-turbo", learning_shots=0, replay=None, search_depth=1, lookahead=1):
        '''
        Search depth = how deep the minimax search tree goes when deciding on a move
        Lookahead = how many moves in the future of minimax play to make before evaluating competing game states
        '''
        self.lookahead = lookahead
        self.minimax = ScoreMinimaxAgent(search_depth)

        return super().__init__(model=model, learning_shots=learning_shots, replay=replay, visualize=True)

    def query(self, env, legal_actions):
        outcomes = self.minimax_outcomes(env, legal_actions)
        return gpt_query.minimax_query(self.model, env, outcomes, self.learning_shots, self.replay)

    def minimax_outcomes(self, env, legal_actions):
        outcomes = {}
        for action in legal_actions:
            action_env, reward, game_over = env.act(action)
            prev_env = env
            for turn in range(self.lookahead):
                if game_over:
                    break
                next_action = self.minimax.choose_action(action_env, prev_env)
                prev_env = action_env
                action_env, reward, game_over = action_env.act(next_action)
            outcomes[action] = (action_env, reward)
        return outcomes



class RandomAgent(ReversiAgent):
    def policy(self, legal_actions, env, _):
        return random.choice(list(legal_actions))

class HeuristicAgent(ReversiAgent):
    def policy(self, legal_actions, env, _):
        best_score = -float('inf')
        player = env.curr_player
        for action in legal_actions:
            test_env, _, _ = env.act(action)
            score = self.heuristic(test_env) * player
            if score > best_score:
                best_score = score
                best_actions = []
            if score == best_score:
                best_actions.append(action)
        return random.choice(best_actions)

    def heuristic(self, env):
        pass

class MinimaxAgent(HeuristicAgent):
    def __init__(self, search_depth):
        self.search_depth = search_depth

    def heuristic(self, env):
        return alphabeta(env, self.search_depth, -float('inf'), float('inf'), self.leaf_heuristic)

    def leaf_heuristic(self, env):
        pass

class ScoreHeuristicAgent(HeuristicAgent):
    def score(self, env):
        s = env.get_score()
        return (s[BLACK] - s[WHITE]) / (s[BLACK] + s[WHITE])

class ScoreGreedyAgent(ScoreHeuristicAgent):
    def heuristic(self, env):
        return self.score(env)

class ScoreMinimaxAgent(ScoreHeuristicAgent, MinimaxAgent):
    def leaf_heuristic(self, env):
        return self.score(env)


class DualAgent(ReversiAgent):
    def __init__(self, agent_b, agent_w):
        self.agents = {BLACK: agent_b, WHITE: agent_w}
        self.agent_b = agent_b
        self.agent_w = agent_w

    def policy(self, legal_actions, env, prev_env):
        return self.agents[env.curr_player].policy(legal_actions, env, prev_env)

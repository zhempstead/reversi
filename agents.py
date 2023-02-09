import random
import time

from alphabeta import alphabeta
from constants import BLACK, WHITE, PLAYER
from display_board import board2str

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

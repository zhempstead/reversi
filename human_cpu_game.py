from lib.agents import RandomAgent, ScoreGreedyAgent, ScoreMinimaxAgent, HumanAgent, DualAgent
from lib.reversi_game import ReversiGame

random = RandomAgent()
greedy = ScoreGreedyAgent()
minimax_3 = ScoreMinimaxAgent(3)
human = HumanAgent()
rg = ReversiGame(8, DualAgent(human, minimax_3), 'replays/human.json', headless=False)
rg.play()

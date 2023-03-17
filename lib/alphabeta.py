from .constants import BLACK

def alphabeta(state, depth, alpha, beta, heuristic):
    if depth == 0:
        return heuristic(state)

    legal_actions = state.legal_actions()
    if not legal_actions:
        legal_actions = [None]

    if state.curr_player == BLACK:
        value = -float('inf')
        for action in legal_actions:
            child, reward, game_over = state.act(action)
            if game_over:
                new_value = reward
            else:
                new_value = alphabeta(child, depth - 1, alpha, beta, heuristic)
            value = max(value, new_value)
            if value > beta:
                break
            alpha = max(alpha, value)
        return value
    else:
        value = float('inf')
        for action in legal_actions:
            child, reward, game_over = state.act(action)
            if game_over:
                new_value = reward
            else:
                new_value = alphabeta(child, depth - 1, alpha, beta, heuristic)
            value = min(value, new_value)
            if value < alpha:
                break
            beta = min(beta, value)
        return value

import random

import chess
import chess.svg
import torch.nn.functional as F


class MCTSNode:
    def __init__(self, game: chess.Board, move=None, parent=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def uct_value(self, total_simulations, bias=1.41):
        if self.visits == 0:
            return float("inf")
        win_ratio = self.wins / self.visits
        exploration = bias * ((total_simulations) ** 0.5 / (1 + self.visits))
        return win_ratio + exploration

    def best_child(self):
        total_simulations = sum([child.visits for child in self.children])
        return max(self.children, key=lambda child: child.uct_value(total_simulations))

    def rollout(self, model):
        pi_logits, value = model(self.game.board_representation())
        return value.item()

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)

    def expand(self, model):
        pi_logits, _ = model(self.game.board_representation())
        pi = F.softmax(pi_logits, dim=0)
        all_possible_moves = [move.uci() for move in chess.Board().legal_moves]
        move_to_index = {move: index for index, move in enumerate(all_possible_moves)}

        for move in self.game.legal_moves:
            move_uci = move.uci()
            if move_uci in move_to_index:
                move_index = move_to_index[move_uci]
                new_state = self.game.copy()
                new_state.push(move)
                move_prob = pi[move_index].item()
                new_node = MCTSNode(new_state, move_prob, move=move, parent=self)
                self.children.append(new_node)


class MCTSNodeSelfPlay:
    def __init__(self, game: chess.Board, move=None, parent=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def uct_value(self, total_simulations, bias=1.41):
        if self.visits == 0:
            return float("inf")
        win_ratio = self.wins / self.visits
        exploration = bias * ((total_simulations) ** 0.5 / (1 + self.visits))
        return win_ratio + exploration

    def best_child(self):
        total_simulations = sum([child.visits for child in self.children])
        return max(self.children, key=lambda child: child.uct_value(total_simulations))

    def rollout(self):
        current_rollout_state = self.game
        while not current_rollout_state.is_game_over():
            possible_moves = list(current_rollout_state.legal_moves)
            action = random.choice(possible_moves)
            current_rollout_state.push(action)
        result = current_rollout_state.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        return 0

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)

    def expand(self):
        for move in self.game.legal_moves:
            new_state = self.game.copy()
            new_state.push(move)
            new_node = MCTSNodeSelfPlay(new_state, move=move, parent=self)
            self.children.append(new_node)


def monte_carlo_tree_search(root, simulations):
    for _ in range(simulations):
        v = tree_policy(root)
        reward = v.rollout()
        v.backpropagate(reward)
    return root.best_child().move


def tree_policy(node):
    while not node.game.is_game_over():
        if len(node.children) == 0:
            node.expand()
            return random.choice(node.children)
        node = node.best_child()
    return node

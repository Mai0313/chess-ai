import datetime
import random

import chess
import chess.svg
import numpy as np
import torch
import torch.nn.functional as F

from src.data.components.convert import ChessConverter
from src.utils.chess_utils import ChessBoard


class MCTSNode:
    def __init__(self, game: chess.Board, move=None, parent=None, model=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.model = model
        self.children = []
        self.wins = 0
        self.visits = 0
        self.converter = ChessConverter()
        self.move_to_index = {}

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
        if self.model:
            board_representation = self.converter.convert_array(self.game)
            tensor_input = torch.from_numpy(board_representation).unsqueeze(0).float().permute(0, 3, 1, 2)
            if next(self.model.parameters()).is_cuda:
                tensor_input = tensor_input.cuda()
            pi_logits, value = self.model(tensor_input)
            return value.item()
        else:
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
        if self.model:
            board_representation = self.converter.convert_array(self.game)
            tensor_input = torch.from_numpy(board_representation).unsqueeze(0).float().permute(0, 3, 1, 2)
            if next(self.model.parameters()).is_cuda:
                tensor_input = tensor_input.cuda()
            pi_logits, _ = self.model(tensor_input)
            pi = F.softmax(pi_logits, dim=0)
            all_possible_moves = [move.uci() for move in self.game.legal_moves]
            self.move_to_index = {move: index for index, move in enumerate(all_possible_moves)}

            for move in self.game.legal_moves:
                move_uci = move.uci()
                if move_uci in self.move_to_index:
                    move_index = self.move_to_index[move_uci]
                    new_state = self.game.copy()
                    new_state.push(move)
                    # move_prob = pi[move_index].item()
                    new_node = MCTSNode(game=new_state, move=move, parent=self, model=self.model)
                    self.children.append(new_node)
        else:
            for move in self.game.legal_moves:
                new_state = self.game.copy()
                new_state.push(move)
                new_node = MCTSNode(new_state, move=move, parent=self)
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


def get_unique_filename(parent_path, prefix, extension):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{parent_path}/{prefix}{current_time}{extension}"


class MCTSModel:
    def __init__(self, model_instance):
        self.data = []
        self.board = chess.Board()
        self.converter = ChessConverter()
        self.model_instance = model_instance

    def self_play(self, show_gui: bool, parent_path: str):
        while not self.board.is_game_over():
            node = MCTSNode(game=self.board, move=None, parent=None, model=self.model_instance)
            move = monte_carlo_tree_search(node, 1000)

            # Save state and MCTS policy
            state = self.converter.convert_array(self.board)
            fen_state = self.board.fen()
            policy = [0] * len(list(self.board.legal_moves))
            for index, child in enumerate(node.children):
                policy[index] = child.visits / node.visits
            self.data.append((state, policy, fen_state))

            self.board.push(move)
            if show_gui:
                ChessBoard().show(self.board, show_gui)

        # Add game result to each step
        result = 0
        if self.board.result() == "1-0":
            result = 1
        elif self.board.result() == "0-1":
            result = -1
        if parent_path:
            self.data = [(state, fen_state, policy, result) for state, policy in self.data]
            states, fen_state, policies, values = zip(*self.data)
            npz_filename = get_unique_filename(parent_path, "data_", ".npz")
            pgn_filename = get_unique_filename(parent_path, "data_", ".pgn")
            np.savez(
                npz_filename,
                states=np.array(states),
                fen_state=np.array(fen_state),
                policies=np.array(policies),
                values=np.array(values),
            )
            # Save as PGN
            game = chess.pgn.Game().from_board(self.board)
            with open(pgn_filename, "w") as pgn_file:
                game.accept(chess.pgn.FileExporter(pgn_file))
        self.reset()
        return state, policy, result

    def reset(self):
        self.data = []
        self.board = chess.Board()

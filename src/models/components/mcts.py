import chess
import numpy as np
import torch


def array_to_board(board_array):
    piece_symbols = {
        0: "p",
        1: "n",
        2: "b",
        3: "r",
        4: "q",
        5: "k",
        6: "P",
        7: "N",
        8: "B",
        9: "R",
        10: "Q",
        11: "K",
    }

    board = chess.Board()
    board.clear_board()

    for rank in range(8):
        for file in range(8):
            for idx, symbol in piece_symbols.items():
                if board_array[7 - rank, file, idx].item() == 1.0:
                    board.set_piece_at(rank * 8 + file, chess.Piece.from_symbol(symbol))
    return board


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.values = []
        self.is_expanded = False

    @property
    def is_terminal(self):
        board = array_to_board(self.state[0].cpu().numpy())
        return board.is_game_over()

    @property
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def expand(self, child_states):
        self.is_expanded = True
        for state in child_states:
            self.children.append(MCTSNode(state, parent=self))

    def best_child(self, c):
        uct_values = [
            (child.value / (child.visits + 1e-10)) + c * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-10))
            for child in self.children
        ]
        return self.children[np.argmax(uct_values)]


def MCTS(root, model, num_simulations, c=1):
    for _ in range(num_simulations):
        leaf = traverse(root)
        simulation_result = rollout(leaf, model)
        backpropagate(leaf, simulation_result)

    return max(root.children, key=lambda node: node.visits).state


def traverse(node):
    while node.is_expanded:
        node = node.best_child(c=1)
    return node


def rollout(node, model):
    pi_logits, value = model(torch.tensor(node.state).float())
    probs = torch.nn.functional.softmax(pi_logits, dim=-1).cpu().numpy()

    if node.is_terminal:
        return -node.value

    valid_actions = list(node.state.legal_moves)

    # 確保 probs 只涵蓋 valid_actions
    probs = probs[np.array([action.uci() for action in valid_actions])]
    probs /= probs.sum()

    chosen_action = np.random.choice(valid_actions, p=probs)

    new_state = node.state.copy()
    new_state.push(chosen_action)

    child_states = [new_state]
    node.expand(child_states)

    return value.item()


def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent

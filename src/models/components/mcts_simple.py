class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        return max(self.children, key=lambda child: child.value / (child.visits + 1e-7))

    def expand(self):
        for new_state in self.state.get_next_states():
            self.children.append(Node(new_state, self))

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


def MCTS(root_state, iterations):
    root = Node(root_state)
    for _ in range(iterations):
        node = root
        while not node.is_leaf():
            node = node.select_child()
        # Expansion
        if not node.state.is_terminal():
            node.expand()
            node = node.children[0]
        reward = node.state.simulate()
        node.backpropagate(reward)
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.state.action

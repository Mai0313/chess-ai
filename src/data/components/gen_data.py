import os
import shutil
import numpy as np
import chess
import chess.pgn
from rich.progress import Progress
from sklearn.model_selection import train_test_split
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class ChessDataLoader:
    def __init__(self):
        pass

    # Function to save data
    def save_data(self, data, labels, file_name):
        np.savez(file_name, data=data, labels=labels)

    # Function to load data
    def load_data(self, file_name):
        loaded_data = np.load(file_name)
        return loaded_data["data"], loaded_data["labels"]

    def merge_saved_data(self, output_filename):
        folder_path = "./data/temp"
        all_data = []
        all_labels = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".npz"):
                file_path = f"{folder_path}/{file_name}"
                data, labels = self.load_data(file_path)
                all_data.extend(data)
                all_labels.extend(labels)

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        self.save_data(all_data, all_labels, output_filename)


class ChessDataGenerator:
    def __init__(self):
        pass

    def __board_to_array(self, board):
        board_array = np.zeros((8, 8, 12), dtype=np.float32)

        piece_idx = {
            "p": 0,
            "P": 6,
            "n": 1,
            "N": 7,
            "b": 2,
            "B": 8,
            "r": 3,
            "R": 9,
            "q": 4,
            "Q": 10,
            "k": 5,
            "K": 11,
        }

        for i in range(8):
            for j in range(8):
                square = 8 * (7 - j) + i  # Calculate square index
                piece = board.piece_at(square)
                if piece:
                    board_array[j, i, piece_idx[piece.symbol()]] = 1.0
        return board_array

    def convert_data_from_realworld(self, input_path, train_cases_path, val_cases_path):
        data = []
        labels = []
        filenames = [f for f in os.listdir(input_path) if f.endswith(".pgn")]

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Converting PGN files from real world...", total=len(filenames)
            )

            for filename in filenames:
                with open(f"{input_path}/{filename}") as pgn:
                    while True:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break  # End of file
                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)
                            board_array = self.__board_to_array(board)
                            label = (
                                1.0 if board.turn == chess.WHITE else 0.0
                            )  # 1 for white's turn, 0 for black
                            data.append(board_array)
                            labels.append(label)

                progress.update(task, advance=1)

        # Adjust dimensions
        data = np.array(data)
        labels = np.array(labels)
        data = np.transpose(data, (0, 3, 1, 2))
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        ChessDataLoader().save_data(X_train, y_train, train_cases_path)
        ChessDataLoader().save_data(X_val, y_val, val_cases_path)

        return X_train, X_val, y_train, y_val

    def generate_data(self, num_games, output_filename):
        data = []
        labels = []
        folder_path = "./data/temp"
        os.makedirs(folder_path, exist_ok=True)
        save_interval = num_games * 0.2

        with Progress() as progress:
            task = progress.add_task("[cyan]Generating data...", total=num_games)
            for i in range(1, num_games + 1):
                board = chess.Board()
                while not board.is_game_over():
                    legal_moves = list(board.legal_moves)
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    board_array = self.__board_to_array(board)
                    label = 1.0 if board.turn == chess.WHITE else 0.0
                    data.append(board_array)
                    labels.append(label)

                progress.update(task, advance=1)

                if i % save_interval == 0:
                    data_array = np.array(data)
                    labels_array = np.array(labels)
                    data_array = np.transpose(data_array, (0, 3, 1, 2))

                    file_name = f"{folder_path}/generated_cases_{i}.npz"
                    ChessDataLoader().save_data(data_array, labels_array, file_name)

                    data = []
                    labels = []
        ChessDataLoader().merge_saved_data(output_filename)
        shutil.rmtree(folder_path)


if __name__ == "__main__":
    input_path = "./data/20230929_raw_data"
    train_cases_path = "./data/train_cases.npz"
    val_cases_path = "./data/val_cases.npz"
    ChessDataGenerator().convert_data_from_realworld(input_path, train_cases_path, val_cases_path)
    ChessDataGenerator().generate_data(30, "./data/test_cases.npz")

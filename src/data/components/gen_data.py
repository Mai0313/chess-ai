import os
import shutil
import numpy as np
import chess
import chess.pgn
from rich.progress import Progress
from sklearn.model_selection import train_test_split
import rootutils
import autorootcwd

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.convert import ChessConverter


class ChessDataLoader:
    def __init__(self):
        pass

    # Function to save data
    def save_data(self, data, labels, fens, stockfish_evals, file_name):
        np.savez(file_name, data=data, labels=labels, fens=fens, stockfish_evals=stockfish_evals)

    # Function to load data
    def load_data(self, file_name):
        loaded_data = np.load(file_name)
        return (
            loaded_data["data"],
            loaded_data["labels"],
            loaded_data["fens"],
            loaded_data["stockfish_evals"],
        )

    def merge_saved_data(self, output_filename):
        folder_path = "./data/temp"
        all_data = []
        all_labels = []
        all_fens = []
        all_stockfish_evals = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".npz"):
                file_path = f"{folder_path}/{file_name}"
                data, labels, fens, stockfish_evals = self.load_data(file_path)
                all_data.extend(data)
                all_labels.extend(labels)
                all_fens.extend(fens)
                all_stockfish_evals.extend(stockfish_evals)

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_fens = np.array(all_fens)
        all_stockfish_evals = np.array(all_stockfish_evals)
        self.save_data(all_data, all_labels, all_fens, all_stockfish_evals, output_filename)


class ChessDataGenerator:
    def __init__(self, eval_option: bool = False):
        stockfish_params = {
            "Debug Log File": "",
            "Contempt": 0,
            "Min Split Depth": 0,
            "Threads": 8,
            "Ponder": "false",
            "Hash": 16,
            "MultiPV": 1,
            "Skill Level": 20,
            "Move Overhead": 10,
            "Minimum Thinking Time": 10,
            "Slow Mover": 100,
            "UCI_Chess960": "false",
            "UCI_LimitStrength": "false",
            "UCI_Elo": 3500,
        }
        self.eval_option = eval_option
        if self.eval_option:
            from stockfish import Stockfish
            self.stockfish = Stockfish(path="engine/stockfish/stockfish-ubuntu-x86-64-avx2", parameters=stockfish_params)

    def get_stockfish_evaluation(self, fen):
        if self.eval_option:
            self.stockfish.set_fen_position(fen)
            eval_info = self.stockfish.get_evaluation()
            if eval_info["type"] == "cp":
                return eval_info["value"] / 100.0  # Convert centipawn to typical value range [-1, 1]
            else:  # If it's a mate score
                return 1.0 if eval_info["value"] > 0 else -1.0
        else:
            return 1.0

    def convert_data(self, input_path, train_cases_path, val_cases_path):
        all_data = []
        all_labels = []
        all_fens = []
        all_stockfish_evals = []

        filenames = [f for f in os.listdir(input_path) if f.endswith(".pgn")]
        with Progress() as progress:
            task = progress.add_task("[cyan]Converting PGN files to npz...", total=len(filenames))
            for filename in filenames:
                data = []
                labels = []
                fens = []
                stockfish_evals = []
                output_filename = filename.replace(".pgn", ".npz")
                try:
                    with open(f"{input_path}/{filename}") as pgn:
                        while True:
                            game = chess.pgn.read_game(pgn)
                            if game is None:
                                break  # End of file
                            board = game.board()
                            for move in game.mainline_moves():
                                board.push(move)
                                board_array = ChessConverter().convert_array(board)
                                fen = board.fen()
                                stockfish_eval = self.get_stockfish_evaluation(fen)
                                # 1 for white's turn, 0 for black's turn
                                label = 1.0 if board.turn == chess.WHITE else 0.0
                                data.append(board_array)
                                labels.append(label)
                                fens.append(fen)
                                stockfish_evals.append(stockfish_eval)
                        progress.update(task, advance=1)
                except UnicodeDecodeError:
                    print(f"Skipping {filename} due to UnicodeDecodeError.")
                    progress.update(
                        task, advance=1, description=f"[cyan]Skipping {filename} due to UnicodeDecodeError."
                    )
                    continue  # Skip to the next iteration

                all_data.extend(data)
                all_labels.extend(labels)
                all_fens.extend(fens)
                all_stockfish_evals.extend(stockfish_evals)

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_fens = np.array(all_fens)
        all_stockfish_evals = np.array(all_stockfish_evals)
        all_data = np.transpose(all_data, (0, 3, 1, 2))
        (
            X_train,
            X_val,
            y_train,
            y_val,
            fens_train,
            fens_val,
            stockfish_evals_train,
            stockfish_evals_val,
        ) = train_test_split(all_data, all_labels, all_fens, all_stockfish_evals, test_size=0.2, random_state=42)
        ChessDataLoader().save_data(X_train, y_train, fens_train, stockfish_evals_train, train_cases_path)
        ChessDataLoader().save_data(X_val, y_val, fens_val, stockfish_evals_val, val_cases_path)
        return X_train, X_val, y_train, y_val

    def generate_data(self, num_games, output_filename):
        data = []
        labels = []
        fens = []
        stockfish_evals = []
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
                    board_array = ChessConverter().convert_array(board)
                    fen = board.fen()
                    stockfish_eval = self.get_stockfish_evaluation(fen)
                    label = 1.0 if board.turn == chess.WHITE else 0.0
                    data.append(board_array)
                    labels.append(label)
                    fens.append(fen)
                    stockfish_evals.append(stockfish_eval)

                progress.update(task, advance=1)

                if i % save_interval == 0:
                    data_array = np.array(data)
                    labels_array = np.array(labels)
                    fens = np.array(fens)
                    stockfish_evals = np.array(stockfish_evals)
                    data_array = np.transpose(data_array, (0, 3, 1, 2))

                    file_name = f"{folder_path}/generated_cases_{i}.npz"
                    ChessDataLoader().save_data(data_array, labels_array, fens, stockfish_evals, file_name)

                    data = []
                    labels = []
                    fens = []
                    stockfish_evals = []
        ChessDataLoader().merge_saved_data(output_filename)
        shutil.rmtree(folder_path)


if __name__ == "__main__":
    input_path = "./data/20230929_raw_data"
    train_cases_path = "./data/train_cases.npz"
    val_cases_path = "./data/val_cases.npz"
    # ChessDataGenerator().convert_data(input_path, train_cases_path, val_cases_path)
    # ChessDataGenerator().convert_data_from_realworld(input_path, train_cases_path, val_cases_path)
    ChessDataGenerator().generate_data(1000, "./data/train_1000_gen_cases.npz")

import os
import shutil
import numpy as np
import chess
import chess.pgn
from rich.progress import Progress
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# Function to save data
def save_data(data, labels, file_name):
    np.savez(file_name, data=data, labels=labels)


# Function to load data
def load_data(file_name):
    loaded_data = np.load(file_name)
    return loaded_data["data"], loaded_data["labels"]


def board_to_array(board):
    # Initialize an empty array to represent the board
    board_array = np.zeros((8, 8, 12), dtype=np.float32)

    # Mapping of piece types to index in the third dimension of the board_array
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


def convert_data_from_realword(path):
    data = []
    labels = []
    filenames = [f for f in os.listdir(path) if f.endswith(".pgn")]
    for filename in filenames:
        pgn = open(f"{path}/{filename}")
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break  # End of file
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                board_array = board_to_array(board)
                label = (
                    1.0 if board.turn == chess.WHITE else 0.0
                )  # 1 for white's turn, 0 for black
                data.append(board_array)
                labels.append(label)

    # Adjust dimensions
    data = np.transpose(data, (0, 3, 1, 2))
    os.makedirs("./data", exist_ok=True)
    save_data(np.array(data), np.array(labels), "./data/real_cases.npz")
    return np.array(data), np.array(labels)

def merge_saved_data():
    folder_path = "./data/temp"
    all_data = []
    all_labels = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npz"):
            file_path = f"{folder_path}/{file_name}"
            data, labels = load_data(file_path)
            all_data.extend(data)
            all_labels.extend(labels)
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    save_data(all_data, all_labels, "./data/generated_cases_merged.npz")

def generate_data(num_games, save_interval):
    data = []
    labels = []
    folder_path = "./data/temp"
    os.makedirs(folder_path, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[cyan]Generating data...", total=num_games)
        for i in range(1, num_games + 1):
            board = chess.Board()
            while not board.is_game_over():
                legal_moves = list(board.legal_moves)
                move = np.random.choice(legal_moves)
                board.push(move)
                board_array = board_to_array(board)
                label = 1.0 if board.turn == chess.WHITE else 0.0
                data.append(board_array)
                labels.append(label)

            progress.update(task, advance=1)

            if i % save_interval == 0:
                data_array = np.array(data)
                labels_array = np.array(labels)
                data_array = np.transpose(data_array, (0, 3, 1, 2))

                file_name = f"{folder_path}/generated_cases_{i}.npz"
                save_data(data_array, labels_array, file_name)

                data = []
                labels = []
    merge_saved_data()
    shutil.rmtree(folder_path)

if __name__ == "__main__":
    convert_data_from_realword("./data/chess_raw")
    generate_data(5000, 1000)

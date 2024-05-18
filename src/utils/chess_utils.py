from typing import Union
import datetime

import chess
import numpy as np
import torch
from pydantic import Field, BaseModel, ConfigDict
import chess.pgn
import chess.svg
from rich.text import Text
import chess.engine
from rich.console import Console
import IPython.display as display

from src.models.components.mcts import MCTSNode, monte_carlo_tree_search


class ChessUtils(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    board: Union[chess.Board, np.ndarray] = Field(..., frozen=True)

    def convert2numpy(self) -> np.ndarray:
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
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                board_array[7 - rank, file, piece_idx[piece.symbol()]] = 1.0
        return board_array

    def show(self, gui: bool) -> None:
        board_svg = chess.svg.board(board=self.board, size=300)
        if gui:
            display.clear_output(wait=True)
            display.display(display.HTML(board_svg))
        else:
            display.clear_output(wait=True)
            print(self.board)  # noqa: T201

    def move(self, model, gpu: bool):
        best_move = None
        best_value = -1.0  # Initialize with a low value
        board = self.board
        for move in board.legal_moves:
            board.push(move)
            board_array = ChessUtils.convert2numpy(board)
            board_array = np.transpose(board_array, (2, 0, 1))
            board_array = torch.tensor(board_array).float().unsqueeze(0)
            if gpu and torch.cuda.is_available():
                board_array = board_array.to("cuda")
            value = model(board_array).item()
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def extract_game_info(
        self, game: chess.pgn.Game, idx: int, current_time: str
    ) -> chess.pgn.Game:
        game.headers["Event"] = "Lc0 v.s. Lc0"
        game.headers["Site"] = idx
        game.headers["Date"] = current_time
        game.headers["White"] = "Lc0_White"
        game.headers["Black"] = "Lc0_Black"
        game.headers["Result"] = self.board.result()
        return game

    def save(self, idx: int, game: chess.pgn.Game, info_list: list[dict], output_dir: str):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        game.headers["Event"] = "Lc0 v.s. Lc0"
        game.headers["Site"] = f"{idx}"
        game.headers["Date"] = current_time
        game.headers["White"] = "Lc0_White"
        game.headers["Black"] = "Lc0_Black"
        game.headers["Result"] = self.board.result()

        output_pgn_filename = f"{output_dir}/lc0_vs_lc0_{current_time}.pgn"
        with open(output_pgn_filename, "w") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)

        output_npz_filename = f"{output_dir}/lc0_vs_lc0_{current_time}.npz"
        np.savez(output_npz_filename, game_info=info_list, game_pgn=game)


class ChessGame:
    def __init__(self, white_model: torch.nn.Module, gpu: bool = False, mcts: bool = False):
        """Initializes the ChessGame class.

        Args:
            white_model (torch.nn.Module): Model to be used when playing as white.
            black_model (torch.nn.Module, optional): Model to be used when playing as black. If not provided, white_model will be used.
            gpu (bool, optional): Flag indicating whether to use GPU. Defaults to False.
            mcts (bool, optional): Flag indicating whether to use Monte Carlo Tree Search.
        """
        self.board = chess.Board()
        self.white_model = white_model
        self.gpu = gpu
        self.mcts = mcts
        if self.gpu and torch.cuda.is_available():
            self.white_model.to("cuda:0")

    def reset_board(self):
        """Resets the game board."""
        self.board.reset()

    def model_move(self):
        if self.mcts:
            node = MCTSNode(self.board, None, None, self.white_model)
            move = monte_carlo_tree_search(node, 1000)
        else:
            move = ChessUtils(board=self.board).move(self.white_model, self.gpu)
        return move

    def self_play(self, gui: bool):
        """The model plays against itself.

        Args:
            gui (bool): Flag indicating whether to show a GUI. Defaults to False.
        """
        while not self.board.is_game_over():
            move = self.model_move()
            self.board.push(move)
            ChessUtils().show(self.board, gui)

    def model_vs_model(self, black_model: torch.nn.Module, gui: bool):
        """Two models play against each other.

        Args:
            black_model (torch.nn.Module): Model to be used when playing as black.
            gui (bool): Flag indicating whether to show a GUI. Defaults to False.
            mcts (bool, optional): Flag indicating whether to use Monte Carlo Tree Search.
        """
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move = self.model_move()
            else:
                if self.gpu and torch.cuda.is_available():
                    black_model.to("cuda:0")
                move = self.model_move()
            self.board.push(move)
            ChessUtils().show(self.board, gui)

    def play_against_ai(self, gui: bool):
        """Play a game against the AI model.

        Args:
            gui (bool): Flag indicating whether to show a GUI.
        """
        while not self.board.is_game_over():
            ChessUtils().show(self.board, gui)
            valid_move = False
            while not valid_move:
                try:
                    human_move = input("Enter your move: ")
                    if human_move in ["q", "quit", "exit"]:
                        return
                    self.board.push_san(human_move)
                    valid_move = True
                    ChessUtils().show(self.board, gui)
                except ValueError:
                    print("Invalid move. Please enter a valid move.")  # noqa: T201
            if not self.board.is_game_over():
                move = self.model_move()
                self.board.push(move)
                print(f"Model's move: {move}")  # noqa: T201
                ChessUtils().show(self.board, gui)

    def model_vs_stockfish(self, gui: bool, stockfish_path: str, cpu_nums: int = 4):
        """Play a game between the model and Stockfish.

        Args:
            gui (bool): Flag indicating whether to show a GUI.
            stockfish_path (str, optional): Path to the Stockfish executable.
                                            Defaults to "../stockfish_linux/stockfish-ubuntu-x86-64-avx2".
            cpu_nums (int, optional): Number of CPUs to be used by Stockfish. Defaults to 4.
        """
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # Set Stockfish to use more threads for increased speed
        engine.configure({"Threads": cpu_nums})

        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move = self.model_move()
            else:
                result = engine.play(self.board, chess.engine.Limit(time=5.0))
                move = result.move
            self.board.push(move)
            ChessUtils().show(self.board, gui)

        engine.quit()

    def model_vs_lc0(self, gui: bool, lc0_path: str, lc0_model_path: str):
        """Play a game between the model and Stockfish.

        Args:
            gui (bool): Flag indicating whether to show a GUI.
            lc0_path (str, optional): Path to the Lc0 executable.
            lc0_model_path (str, optional): Path to the Lc0 model.
        """
        engine = chess.engine.SimpleEngine.popen_uci(lc0_path)

        # Set Stockfish to use more threads for increased speed
        engine.configure({"WeightsFile": lc0_model_path})

        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move = self.model_move()
            else:
                result = engine.play(self.board, chess.engine.Limit(time=5.0))
                move = result.move
            self.board.push(move)
            ChessUtils().show(self.board, gui)

        engine.quit()

    def solve_puzzle(self, board: chess.Board, gui: bool) -> chess.Move:
        """Solve the chess puzzle and return the best move.

        Args:
            board (chess.Board): The current chess board state.
            gui (bool): Flag indicating whether to show a GUI.

        Returns:
            chess.Move: The best move for the given board state.
        """
        move = self.model_move()
        board.push(move)
        ChessUtils().show(board, gui)
        return move

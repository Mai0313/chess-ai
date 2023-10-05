import multiprocessing as mp
from typing import Optional

import chess
import chess.engine
import chess.svg
import IPython.display as display
import numpy as np
import torch


class ChessData:
    def __init__(self):
        pass

    def convert(self, board: chess.Board) -> np.ndarray:
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
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                board_array[7 - rank, file, piece_idx[piece.symbol()]] = 1.0
        return board_array


class ChessBoard:
    def __init__(self):
        pass

    def __check_game_result(self, board: chess.Board):
        """Checks the result of the game."""
        if board.is_checkmate():
            if board.turn == chess.BLACK:  # White made the last move
                print("White wins by checkmate!")
            else:
                print("Black wins by checkmate!")
        elif board.is_stalemate():
            print("The game is a draw due to stalemate!")
        elif board.is_insufficient_material():
            print("The game is a draw due to insufficient material!")
        elif board.can_claim_fifty_moves():
            print("The game is a draw due to the fifty-move rule!")
        elif board.can_claim_threefold_repetition():
            print("The game is a draw due to threefold repetition!")

    def show(self, board: np.ndarray, gui: bool):
        board_svg = chess.svg.board(board=board, size=300)
        if gui:
            display.clear_output(wait=True)
            display.display(display.HTML(board_svg))
        else:
            display.clear_output(wait=True)
            print(board)
        self.__check_game_result(board)


class ChessMove:
    def __init__(self):
        pass

    def move(self, model, board, gpu: bool):
        best_move = None
        best_value = -1.0  # Initialize with a low value
        for move in board.legal_moves:
            board.push(move)
            board_array = ChessData().convert(board)
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


class ChessGame:
    def __init__(
        self,
        white_model: torch.nn.Module,
        gpu: bool = False,
    ):
        """Initializes the ChessGame class.

        Args:
            white_model (torch.nn.Module): Model to be used when playing as white.
            black_model (torch.nn.Module, optional): Model to be used when playing as black.
                                                     If not provided, white_model will be used.
            gpu (bool, optional): Flag indicating whether to use GPU. Defaults to False.
        """
        self.board = chess.Board()
        self.white_model = white_model
        self.gpu = gpu
        if self.gpu and torch.cuda.is_available():
            self.white_model.to("cuda:0")

    def reset_board(self):
        """Resets the game board."""
        self.board.reset()

    def self_play(self, gui: bool):
        """The model plays against itself.

        Args:
            gui (bool): Flag indicating whether to show a GUI. Defaults to False.
        """
        while not self.board.is_game_over():
            move = ChessMove().move(self.white_model, self.board, self.gpu)
            self.board.push(move)
            ChessBoard().show(self.board, gui)

    def model_vs_model(self, black_model: torch.nn.Module, gui: bool):
        """Two models play against each other.

        Args:
            black_model (torch.nn.Module): Model to be used when playing as black.
            gui (bool): Flag indicating whether to show a GUI. Defaults to False.
        """
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move = ChessMove().move(self.white_model, self.board, self.gpu)
            else:
                if self.gpu and torch.cuda.is_available():
                    black_model.to("cuda:0")
                move = ChessMove().move(black_model, self.board, self.gpu)
            self.board.push(move)
            ChessBoard().show(self.board, gui)

    def play_against_ai(self, gui: bool):
        """Play a game against the AI model.

        Args:
            gui (bool): Flag indicating whether to show a GUI.
        """
        while not self.board.is_game_over():
            ChessBoard().show(self.board, gui)
            valid_move = False
            while not valid_move:
                try:
                    human_move = input("Enter your move: ")
                    if human_move in ["q", "quit", "exit"]:
                        return
                    self.board.push_san(human_move)
                    valid_move = True
                    ChessBoard().show(self.board, gui)
                except ValueError:
                    print("Invalid move. Please enter a valid move.")
            if not self.board.is_game_over():
                model_move = ChessMove().move(self.white_model, self.board, self.gpu)
                self.board.push(model_move)
                print(f"Model's move: {model_move}")
                ChessBoard().show(self.board, gui)

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
                move = ChessMove().move(self.white_model, self.board, self.gpu)
            else:
                result = engine.play(self.board, chess.engine.Limit(time=5.0))
                move = result.move
            self.board.push(move)
            ChessBoard().show(self.board, gui)

        engine.quit()

    def solve_puzzle(self, board: chess.Board, gui: bool) -> chess.Move:
        """Solve the chess puzzle and return the best move.

        Args:
            board (chess.Board): The current chess board state.
            gui (bool): Flag indicating whether to show a GUI.

        Returns:
            chess.Move: The best move for the given board state.
        """
        best_move = ChessMove().move(self.white_model, board, self.gpu)
        board.push(best_move)
        ChessBoard().show(board, gui)
        return best_move

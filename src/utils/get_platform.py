import platform


def get_platform():
    lc0_path = "engine/Lc0/lc0"
    stockfish_path = "engine/stockfish/stockfish-ubuntu-x86-64-avx2"
    if platform.system() == "Windows":
        lc0_path = "engine/Lc0/lc0.exe"
        stockfish_path = "engine/stockfish_win/stockfish-windows-x86-64-avx2.exe"
    return lc0_path, stockfish_path


lc0_path, stockfish_path = get_platform()

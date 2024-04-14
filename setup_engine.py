import os
import tarfile
import zipfile
import platform
import subprocess
import urllib.request

from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress

console = Console()
progress = Progress(console=console)


def execute_command(command, error_message):
    result = subprocess.call(command, shell=True)
    if result != 0:
        console.print(Panel.fit(Text(error_message, style="bold red")))
    return result


def download_with_progress(url, filename):
    task_id = progress.add_task("[cyan]Downloading...", total=100)

    def progress_update(count, block_size, total_size):
        percent_complete = (count * block_size * 100) / total_size
        progress.update(task_id, completed=percent_complete)

    urllib.request.urlretrieve(url, filename, reporthook=progress_update)
    progress.stop()


def install_requirements():
    console.print(Panel.fit(Text("Removing Cache...", style="bold green")))
    execute_command("pip cache purge", "There is no Cache to remove")
    console.print(Panel.fit(Text("Installing Requirements ...", style="bold green")))
    with open("requirements.txt") as f:
        packages = f.readlines()

    with Progress() as progress:
        task1 = progress.add_task("[cyan]Installing...", total=len(packages))
        for package in packages:
            package = package.strip()
            progress.update(task1, description=f"[cyan]Installing {package}...")
            result = subprocess.call(
                ["pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if result == 0:
                progress.update(task1, advance=1)
            # else:
            #     console.print(Panel.fit(Text(f"Package: {package} Install Failed", style="bold red")))


def change_dir():
    os.makedirs("engine", exist_ok=True)
    os.chdir("engine")
    console.print("[bold green]Starting the download and extraction process...[/bold green]")


def setup_windows():
    win_stockfish_url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-windows-x86-64-avx2.zip"
    win_stockfish_filename = "stockfish-windows-x86-64-avx2.zip"
    download_with_progress(win_stockfish_url, win_stockfish_filename)

    console.print("[bold cyan]Extracting Stockfish for Windows...[/bold cyan]")
    with zipfile.ZipFile(win_stockfish_filename, "r") as zip_ref:
        zip_ref.extractall(".")
    os.rename("stockfish", "stockfish_win")
    os.remove(win_stockfish_filename)

    lc0_win_url = "https://github.com/LeelaChessZero/lc0/releases/download/v0.30.0/lc0-v0.30.0-windows-gpu-nvidia-cuda.zip"
    lc0_win_filename = "lc0-v0.30.0-windows-gpu-nvidia-cuda.zip"
    os.makedirs("lc0_win", exist_ok=True)
    download_with_progress(lc0_win_url, lc0_win_filename)

    console.print("[bold cyan]Extracting lc0 for Windows...[/bold cyan]")
    with zipfile.ZipFile(lc0_win_filename, "r") as zip_ref:
        zip_ref.extractall("lc0_win")
    os.remove(lc0_win_filename)


def setup_linux():
    linux_stockfish_url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar"
    linux_stockfish_filename = "stockfish-ubuntu-x86-64-avx2.tar"
    download_with_progress(linux_stockfish_url, linux_stockfish_filename)

    console.print("[bold cyan]Extracting Stockfish for Linux...[/bold cyan]")
    with tarfile.open(linux_stockfish_filename, "r") as tar_ref:
        tar_ref.extractall(".")
    os.remove(linux_stockfish_filename)

    console.print("[bold cyan]Cloning and building lc0 for Linux...[/bold cyan]")
    if (
        execute_command(
            "git clone https://github.com/LeelaChessZero/lc0.git lc0_src",
            "Failed to clone lc0 for Linux",
        )
        == 0
    ):
        os.chmod("./lc0_src/build.sh", 0o755)
        if execute_command("./lc0_src/build.sh", "Failed to build lc0 for Linux") == 0:
            os.rename("./lc0_src/build/release", "lc0")
            os.system("rm -rf lc0_src")


def get_model():
    os.makedirs("lc0_model", exist_ok=True)
    model_urls = [
        "https://storage.lczero.org/files/networks-contrib/t2-768x15x24h-swa-5230000.pb.gz",
        "https://storage.lczero.org/files/networks-contrib/t1-512x15x8h-distilled-swa-3395000.pb.gz",
        "https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz",
    ]

    for model_url in model_urls:
        filename = model_url.split("/")[-1]
        download_with_progress(model_url, filename)
        console.print(f"[bold cyan]Extracting {filename}...[/bold cyan]")
        os.system(f"gunzip {filename}")
        os.rename(filename[:-3], "lc0_model/" + filename[:-3])

    console.print("[bold green]Process completed successfully![/bold green]")


if __name__ == "__main__":
    install_requirements()
    change_dir()
    if platform.system() == "Windows":
        setup_windows()
    else:
        setup_linux()
    get_model()

@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: 创建 engine 目录并切换至该目录
IF NOT EXIST "engine" (
    mkdir "engine"
)
cd engine

:: 下载和解压 Stockfish
curl -L "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-windows-x86-64-avx2.zip" -o stockfish.zip
powershell -command "Expand-Archive -LiteralPath 'stockfish.zip' -DestinationPath '.'"
del "stockfish.zip"

:: 下载和解压 Lc0
curl -L "https://github.com/LeelaChessZero/lc0/releases/download/v0.30.0/lc0-v0.30.0-windows-gpu-nvidia-cuda.zip" -o lc0.zip
powershell -command "Expand-Archive -LiteralPath 'lc0.zip' -DestinationPath 'Lc0_win'"
del "lc0.zip"

:: 下载和解压 Leela Chess Zero 模型
curl -L "https://storage.lczero.org/files/networks-contrib/t2-768x15x24h-swa-5230000.pb.gz" -o model1.pb.gz
powershell -command "Expand-Archive -LiteralPath 'model1.pb.gz' -DestinationPath '.'"
del "model1.pb.gz"

curl -L "https://storage.lczero.org/files/networks-contrib/t1-512x15x8h-distilled-swa-3395000.pb.gz" -o model2.pb.gz
powershell -command "Expand-Archive -LiteralPath 'model2.pb.gz' -DestinationPath '.'"
del "model2.pb.gz"

curl -L "https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz" -o model3.pb.gz
powershell -command "Expand-Archive -LiteralPath 'model3.pb.gz' -DestinationPath '.'"
del "model3.pb.gz"

powershell -command "$gzipStream = [System.IO.Compression.GzipStream]::new([System.IO.File]::OpenRead('model1.pb.gz'), [System.IO.Compression.CompressionMode]::Decompress); $targetFileStream = [System.IO.File]::Create('Lc0_win\model1.pb'); $gzipStream.CopyTo($targetFileStream); $gzipStream.Dispose(); $targetFileStream.Dispose()"
powershell -command "$gzipStream = [System.IO.Compression.GzipStream]::new([System.IO.File]::OpenRead('model2.pb.gz'), [System.IO.Compression.CompressionMode]::Decompress); $targetFileStream = [System.IO.File]::Create('Lc0_win\model2.pb'); $gzipStream.CopyTo($targetFileStream); $gzipStream.Dispose(); $targetFileStream.Dispose()"
powershell -command "$gzipStream = [System.IO.Compression.GzipStream]::new([System.IO.File]::OpenRead('model3.pb.gz'), [System.IO.Compression.CompressionMode]::Decompress); $targetFileStream = [System.IO.File]::Create('Lc0_win\model3.pb'); $gzipStream.CopyTo($targetFileStream); $gzipStream.Dispose(); $targetFileStream.Dispose()"

:: 将模型文件移动到 Lc0_win 目录
move "*.pb" "Lc0_win"

echo All tasks completed successfully.
pause

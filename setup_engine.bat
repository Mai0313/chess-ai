@echo off

mkdir engine
pushd engine

powershell -command "Invoke-WebRequest -Uri https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-windows-x86-64-avx2.zip -OutFile stockfish-windows-x86-64-avx2.zip"
tar -xf stockfish-windows-x86-64-avx2.zip
ren stockfish.exe stockfish_win.exe
del stockfish-windows-x86-64-avx2.zip

mkdir Lc0_win
powershell -command "Invoke-WebRequest -Uri https://github.com/LeelaChessZero/lc0/releases/download/v0.30.0/lc0-v0.30.0-windows-gpu-nvidia-cuda.zip -OutFile lc0-v0.30.0-windows-gpu-nvidia-cuda.zip"
tar -xf lc0-v0.30.0-windows-gpu-nvidia-cuda.zip -C Lc0_win
del lc0-v0.30.0-windows-gpu-nvidia-cuda.zip

mkdir lc0_model

powershell -command "Invoke-WebRequest -Uri https://storage.lczero.org/files/networks-contrib/t2-768x15x24h-swa-5230000.pb.gz -OutFile t2-768x15x24h-swa-5230000.pb.gz"
powershell -command "Expand-Archive -Path t2-768x15x24h-swa-5230000.pb.gz -DestinationPath ."
del t2-768x15x24h-swa-5230000.pb.gz

powershell -command "Invoke-WebRequest -Uri https://storage.lczero.org/files/networks-contrib/t1-512x15x8h-distilled-swa-3395000.pb.gz -OutFile t1-512x15x8h-distilled-swa-3395000.pb.gz"
powershell -command "Expand-Archive -Path t1-512x15x8h-distilled-swa-3395000.pb.gz -DestinationPath ."
del t1-512x15x8h-distilled-swa-3395000.pb.gz

powershell -command "Invoke-WebRequest -Uri https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz -OutFile t1-256x10-distilled-swa-2432500.pb.gz"
powershell -command "Expand-Archive -Path t1-256x10-distilled-swa-2432500.pb.gz -DestinationPath ."
del t1-256x10-distilled-swa-2432500.pb.gz

move *.pb lc0_model

echo Done!
pause

popd

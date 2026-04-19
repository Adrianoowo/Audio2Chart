@echo off
echo ====================================================
echo Phase 1: Deep Dataset Extractor ^& Mel-Spectrogram Matrix
echo ====================================================
echo Commencing AI translation of ~3000 songs from D:\Charts
echo (This could take a massive amount of time/RAM!)
python preprocessing.py "D:\Charts"

echo.
echo ====================================================
echo Phase 2: PyTorch Model Training Loop
echo ====================================================
echo Booting neural network to cook the GPU on data.pkl..
python train.py --epochs 30

echo.
echo All Done! Model is now saved in checkpoints.
pause

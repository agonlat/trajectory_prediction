#!/bin/bash
# Shell script to automatically start Social-LSTM training

# Path to Python (adjust if Python is not in PATH)
pythonPath="python"

# Path to train.py
trainScript="./social-lstm/train.py"

# Parameters for training
useCuda="--use_cuda"
seqLength=20
predLength=12
batchSize=5
numEpochs=30
learningRate=0.003
gruFlag=""  # set "--gru" if you want GRU instead of LSTM

# Path to prepared dataset file
datasetFile="--dataset ./data/sgan_format/all_videos.txt"

# Start training
echo "Starting Social-LSTM training..."
$pythonPath $trainScript $useCuda --seq_length $seqLength --pred_length $predLength --batch_size $batchSize --num_epochs $numEpochs --learning_rate $learningRate $gruFlag $datasetFile

echo "Training finished."

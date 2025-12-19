#!/bin/bash

#################################################
## ESSENTIAL LOOPING CODE FOR JUPYTER NOTEBOOK ##
#################################################

# Before the loop starts, set up the Python virtual environment and install dependencies
venv_path="./venv" # Update this to your preferred venv location

# Create virtual environment if it doesn't exist
if [ ! -d "$venv_path" ]; then
    python3 -m venv "$venv_path"
    echo "Virtual environment created at $venv_path."
    #Install dependencies from requirements.txt
    requirements_path="./requirements.txt" # Update this to the path to your requirements.txt file
    pip install -r "$requirements_path"
    echo "Dependencies installed from $requirements_path."
fi

# Activate virtual environment
source "$venv_path/bin/activate"
# source ./.bashrc

# Login to Weights & Biases
wandb login $WANDB_API_KEY

# Define arguments
dataset='ml_100k'
model_type='user'
is_weighted_implicit="no"  # or "no", depending on your requirement

# Run the Python script with appropriate arguments
python -u autoencoder_optimization.py --dataset $dataset --model_type $model_type --is_weighted_implicit $is_weighted_implicit

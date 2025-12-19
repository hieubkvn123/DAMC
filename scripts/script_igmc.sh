#!/bin/bash

#################################################
## SHELL SCRIPT TO RUN JOINT MODEL TRAINING    ##
#################################################

# Set up the Python virtual environment and install dependencies
venv_path="./venv" # Path to your virtual environment

# Create virtual environment if it doesn't exist
if [ ! -d "$venv_path" ]; then
    python3 -m venv "$venv_path"
    echo "Virtual environment created at $venv_path."

    # Install dependencies from requirements.txt
    requirements_path="./requirements.txt" # Path to your requirements.txt file
    "$venv_path/bin/pip" install -r "$requirements_path"
    echo "Dependencies installed from $requirements_path."
fi

# Activate virtual environment
source "$venv_path/bin/activate"

# Login to Weights & Biases
wandb login $WANDB_API_KEY

# Define fixed arguments for the JointModel script
dataset='ml_100k' # Example dataset name
p_values=(0.0)       # Example p_value
model='IGMC'
step=10000         # Example step size
lr=0.01

lambdas=(0 0.01 0.1 1 10 100 1000 10000)

# Loop over latent_side_info and lambda_ values
for p_value in "${p_values[@]}"
    do
        for lambda_ in "${lambdas[@]}"
        do
            # Run the Python script with the defined arguments
            python main_igmc.py --dataset $dataset --model $model --p_value $p_value --step $step --lr $lr --lambda_ $lambda_
        done
    done
# Deactivate virtual environment
deactivate

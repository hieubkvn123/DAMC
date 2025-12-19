#!/bin/bash

#################################################
## SHELL SCRIPT TO RUN SOFT IMPUTE TRAINING    ##
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

# Define fixed arguments for the Soft Impute script
dataset='ml_100k' # Example dataset name
p_values=(0.1)       # Example p_value
max_iter=1000      # Example max iterations
lr=0.1            # Example learning rate

# Define arrays for max_rank and lambda_ values to loop over
max_ranks=(2 4 8)
lambdas=(0.1 1 10 100 1000 5000)

# Loop over max_rank and lambda_ values
for p_value in "${p_values[@]}"
do
    for max_rank in "${max_ranks[@]}"
    do
        for lambda_ in "${lambdas[@]}"
        do
            echo "Running Soft Impute with max_rank: $max_rank, lambda_: $lambda_"

            # Run the Python script with the defined arguments
            python soft_impute_main.py --dataset $dataset --max_rank $max_rank --lambda_ $lambda_ --step $max_iter --lr $lr --p $p_value

            echo "Completed Soft Impute with max_rank: $max_rank, lambda_: $lambda_"
        done
    done
done

# Deactivate virtual environment
deactivate

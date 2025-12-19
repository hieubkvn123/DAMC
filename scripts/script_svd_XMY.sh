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

# Define fixed arguments for the SVD_XMY script
dataset='ml_100k' # Example dataset name
p_values=(0.7)    # Example p_value
model='SVD_XMY'   # Updated model name
step=7000         # Example step size
lr_M=0.01         # Learning rate for M

latent_dims=(4 8)  # Latent dimensions for SVD
lambdas=(1000)     # Regularization strengths
#alphas=(1500)      # Example alpha values

# Loop over latent_dim and lambda_ values
for p_value in "${p_values[@]}"
do
    for latent_dim in "${latent_dims[@]}"
    do
        for lambda_ in "${lambdas[@]}"
        do
            echo "Running SVD_XMY model with latent_dim: $latent_dim, lambda_: $lambda_, alpha: $alpha, p_value: $p_value"

            # Run the Python script with the defined arguments
            python main_svd_xmy.py --dataset $dataset --p_value $p_value --step $step --lr_M $lr_M --latent_dim $latent_dim --lambda_ $lambda_

            echo "Completed SVD_XMY model with latent_dim: $latent_dim, lambda_: $lambda_, alpha: $alpha, p_value: $p_value"
        done
    done
done

# Deactivate virtual environment
deactivate
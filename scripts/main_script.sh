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
model='JointModel'
step=10000         # Example step size
lr_M=0.1
lr_ae=0.1
#is_weighted_implicit='--is_weighted_implicit' # Use flag for weighted implicit matrix

# Define arrays for latent_side_info and lambda_ values to loop over
#num_layers=(2)
latent_side_infos=(4 8)
# latent_Ms=(2 4 8)
lambdas=(0 100 1000 10000)

# Loop over latent_side_info and lambda_ values
for p_value in "${p_values[@]}"
    do
    for latent_side_info in "${latent_side_infos[@]}"
    do
        # for latent_M in "${latent_Ms[@]}"
        # do
            for lambda_ in "${lambdas[@]}"
            do
                echo "Running model with latent_side_info: $latent_side_info, lambda_: $lambda_"

                # Run the Python script with the defined arguments
                python main.py --dataset $dataset --model $model --p_value $p_value --step $step --lr_M $lr_M --lr_ae $lr_ae --latent_side_info $latent_side_info --latent_M $latent_side_info --lambda_ $lambda_ #--is_weighted_implicit $is_weighted_implicit

                echo "Completed model with latent_side_info: $latent_side_info, lambda_: $lambda_"
            # done
        done
    done
done
# Deactivate virtual environment
deactivate

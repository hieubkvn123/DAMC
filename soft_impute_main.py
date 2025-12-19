import argparse
import wandb
import numpy as np
import tensorflow as tf

from models.soft_impute import SoftImpute  # Assuming SoftImpute is in a separate file named soft_impute.py
from utils.data_utils import prepare_data  # Assuming this function exists in your utils module

def main(args):
    # Extract arguments
    dataset_name = args.dataset
    max_rank = args.max_rank
    lambda_ = args.lambda_
    step = args.step
    lr = args.lr
    p = args.p

    # Prepare data
    R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test = prepare_data(dataset_name, p)

    # Initialize wandb
    wandb.init(project='SoftImpute', 
               name=f'SoftImpute-{dataset_name}-lambda={lambda_}-max_rank={max_rank}',
               entity="munchong915",  # Replace with your WandB account name
               config={
                   'dataset': dataset_name,
                   'max_rank': max_rank,
                   'lambda_': lambda_,
                   'step': step,
                   'lr': lr,
                   'p': p,
               })

    # Instantiate Soft Impute
    soft_impute = SoftImpute(max_rank=wandb.config.max_rank, 
                             lambda_=wandb.config.lambda_,
                             step=wandb.config.step,
                             lr=wandb.config.lr,
                             p=wandb.config.p)

    # Fit the model
    train_rmse, val_rmse, test_rmse = soft_impute.fit(R_train, R_val, R_test)

    wandb.finish()
    
    return train_rmse, val_rmse, test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for training")
    parser.add_argument("--max_rank", type=int, required=True, help="Maximum rank for Soft Impute")
    parser.add_argument("--lambda_", type=float, required=True, help="Regularization parameter for Soft Impute")
    parser.add_argument("--step", type=int, required=True, help="Number of steps for training")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for Soft Impute")
    parser.add_argument("--p", type=float, required=True, help="Percentage of data to keep")
    # parser.add_argument('--init_with_svd', action='store_true', help='Initialize with SVD if set, random otherwise')
    # Add other arguments as needed

    args = parser.parse_args()
    train_rmse, val_rmse, test_rmse = main(args)

    print(f"RMSE - Train: {train_rmse}, Validation: {val_rmse}, Test: {test_rmse}")

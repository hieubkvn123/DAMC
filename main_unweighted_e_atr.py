import argparse
import wandb
import tensorflow as tf

from models.unweighted_e_atr import Unweighted_E_ATR
from utils.data_utils import prepare_data  # Assuming this function exists in your utils module

def main(args):
    # Extract arguments
    dataset_name = args.dataset
    p_value = args.p_value
    lr_M = args.lr_M
    lr_ae = args.lr_ae
    step = args.step
    latent_side_info = args.latent_side_info
    latent_M = args.latent_M
    lambda_ = args.lambda_

    # Prepare data
    R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test = prepare_data(dataset_name, p_value)

    # Initialize wandb
    wandb.init(project='Unweighted_E_ATR_2', 
               name=f'Unweighted_E_ATR-dataset={dataset_name}, p={p_value}, Latent dim: {latent_side_info}, Lambda_: {lambda_}',
               entity="munchong915",  # Replace with your WandB account name
               config={
                   'dataset': dataset_name,
                   'p_value': p_value,
                   'latent_side_info': latent_side_info,
                   'latent_M': latent_M,
                   'lambda_M': lambda_,
                   'step': step,
                   'lr_M': lr_M,
                   'lr_ae': lr_ae,
                   'embeddings_epochs': 200,
                   'droputout': 0.2,
                   'batch_norm': True,
               })

    model = Unweighted_E_ATR(wandb.config, dataset_name)
    
    # Fit the model
    best_train_rmse, best_val_rmse, best_test_rmse = model.fit(R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test)

    wandb.finish()
    
    return best_train_rmse, best_val_rmse, best_test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for training")
    parser.add_argument("--p_value", type=float, required=True, help="Percentage of explicit data to be removed")
    parser.add_argument("--lr_M", type=float, required=True, help="Learning rate of U and V")
    parser.add_argument("--lr_ae", type=float, required=True, help="Learning rate of user/item autoencoders")
    parser.add_argument("--step", type=int, required=True, help="Step size")
    parser.add_argument("--latent_side_info", type=int, required=True, help="Latent Side Info")
    parser.add_argument("--latent_M", type=int, required=True, help="Latent M")
    parser.add_argument("--lambda_", type=float, required=True, help="Lambda parameter")
    # Add other arguments as needed

    args = parser.parse_args()
    best_train_rmse, best_val_rmse, best_test_rmse = main(args)

    print(f"Best RMSE - Train: {best_train_rmse}, Validation: {best_val_rmse}, Test: {best_test_rmse}")

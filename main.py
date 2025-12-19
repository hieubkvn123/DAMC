import argparse
import wandb
import tensorflow as tf

from models.joint_model import JointModel
from utils.data_utils import prepare_data  # Assuming this function exists in your utils module

def main(args):
    # Extract arguments
    dataset_name = args.dataset
    p_value = args.p_value
    model_name = args.model
    lr_M = args.lr_M
    lr_ae = args.lr_ae
    step = args.step
    latent_side_info = args.latent_side_info
    latent_M = args.latent_M
    lambda_ = args.lambda_
    is_weighted_implicit = args.is_weighted_implicit
    #num_layer = args.num_layer

    # Prepare data
    R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test = prepare_data(dataset_name, p_value)

    # Initialize wandb
    wandb.init(project='Weighted_XMY_3', 
               name=f'{model_name}-dataset={dataset_name}, p={p_value}, Latent dim: {latent_side_info}, Lambda_: {lambda_}, is_weighted_implicit: {is_weighted_implicit}',
               entity="munchong915",  # Replace with your WandB account name
               config={
                   'dataset': dataset_name,
                   'p_value': p_value,
                   'latent_dim': latent_side_info,
                   'M_max_rank': latent_M,
                   'lambda_M': lambda_,
                   'step': step,
                   #'num_layer': num_layer,
                   'lr_M': lr_M,
                   'lr_ae': lr_ae,
                   'embeddings_epochs': 100,
                   'is_weighted_implicit': is_weighted_implicit,
                   'droputout': 0.2,
                   'batch_norm': True,
               })

    # Model selection
    if model_name == 'JointModel':
        model = JointModel(wandb.config, dataset_name, 'Weighted_XMY')
    else:
        raise ValueError(f"Model '{model_name}' not recognized")

    # Fit the model
    best_train_rmse, best_val_rmse, best_test_rmse = model.fit(R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test)

    wandb.finish()
    
    return best_train_rmse, best_val_rmse, best_test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for training")
    parser.add_argument("--model", type=str, required=True, choices=['JointModel'], help="Model to train")  # Update choices as new models are added
    parser.add_argument("--p_value", type=float, required=True, help="Percentage of explicit data to be removed")
    parser.add_argument("--lr_M", type=float, required=True, help="Learning rate of U and V")
    parser.add_argument("--lr_ae", type=float, required=True, help="Learning rate of user/item autoencoders")
    parser.add_argument("--step", type=int, required=True, help="Step size")
    parser.add_argument("--latent_side_info", type=int, required=True, help="Latent Side Info")
    parser.add_argument("--latent_M", type=int, required=True, help="Latent M")
    parser.add_argument("--lambda_", type=float, required=True, help="Lambda parameter")
    parser.add_argument('--is_weighted_implicit', action='store_true', help='Use weighted implicit matrix')
    # Add other arguments as needed

    args = parser.parse_args()
    best_train_rmse, best_val_rmse, best_test_rmse = main(args)

    print(f"Best RMSE - Train: {best_train_rmse}, Validation: {best_val_rmse}, Test: {best_test_rmse}")

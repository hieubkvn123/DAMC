import argparse
import wandb

# Assuming SVD_XMY is in models/svd_xmy.py
from models.svd_xmy import SVD_XMY
from utils.data_utils import prepare_data  # Ensure this is compatible with your dataset and model requirements

def main(args):
    dataset_name = args.dataset
    p_value = args.p_value
    lr_M = args.lr_M
    step = args.step
    latent_side_info = args.latent_side_info
    latent_M = args.latent_M
    lambda_ = args.lambda_

    # Prepare data
    print('Preparing dataset...')
    R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test = prepare_data(dataset_name, p_value)
    print('Dataset prepared.')

    # Initialize wandb
    wandb.init(project='SVD_XMY_Project',
               name=f'SVD_XMY-{args.dataset}-latent_side_info={args.latent_side_info}-lambda_M={args.lambda_M}',
               entity='munchong915',  # Replace with your WandB entity name
               config={
                   'dataset': dataset_name,
                   'p_value': p_value,
                   'latent_side_info': latent_side_info,
                   'latent_M': latent_M,
                   'lambda_M': lambda_,
                   'step': step,
                   'lr_M': lr_M,
               })

    # Initialize the model with the configuration
    model = SVD_XMY(wandb.config)

    # Fit the model with the prepared data
    best_train_rmse, best_val_rmse, best_test_rmse = model.fit(R_bar_train, R_train, R_val, R_test)

    # Finish the wandb run
    wandb.finish()

    return best_train_rmse, best_val_rmse, best_test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SVD_XMY model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for training.")
    parser.add_argument("--p_value", type=float, required=True, help="Percentage of explicit data to be removed.")
    parser.add_argument("--latent_side_info", type=int, required=True, help="Dimensionality of the latent feature space, equivalent to latent_dim.")
    parser.add_argument("--latent_M", type=int, required=True, help="This parameter will be ignored in SVD_XMY setup but kept for compatibility.")
    parser.add_argument("--lambda_", type=float, required=True, help="Regularization strength for U and V matrices.")
    parser.add_argument("--step", type=int, required=True, help="Number of training steps.")
    parser.add_argument("--lr_M", type=float, required=True, help="Learning rate for U and V matrices optimization.")

    args = parser.parse_args()
    best_train_rmse, best_val_rmse, best_test_rmse = main(args)

    print(f"Best RMSE - Train: {best_train_rmse}, Validation: {best_val_rmse}, Test: {best_test_rmse}")

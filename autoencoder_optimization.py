import argparse
import optuna
import math
import tensorflow as tf
import wandb
from autoencoders.autoencoder import UserAutoEncoder, ItemAutoEncoder
from autoencoders.weighted_autoencoder import WeightedUserAutoEncoder, WeightedItemAutoEncoder, CustomLoss
from utils.weights_utils import MatrixProcessor
from utils.data_utils import preprocess_data
from utils.evaluation import calculate_recall_at_k
from tensorflow.keras.callbacks import EarlyStopping

class WandbLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Initialize an empty dictionary to hold the metrics to log
            metrics_to_log = {}

            # Check for 'mse' and 'val_mse' in logs and calculate RMSE
            if 'mse' in logs:
                metrics_to_log['mse'] = logs['mse']
                metrics_to_log['rmse'] = math.sqrt(logs['mse'])  # Calculate RMSE

            if 'val_mse' in logs:
                metrics_to_log['val_mse'] = logs['val_mse']
                metrics_to_log['val_rmse'] = math.sqrt(logs['val_mse'])  # Calculate validation RMSE

            # Check for 'loss' and 'val_loss'
            if 'loss' in logs:
                metrics_to_log['loss'] = logs['loss']
            if 'val_loss' in logs:
                metrics_to_log['val_loss'] = logs['val_loss']

            # Log the metrics using wandb
            wandb.log(metrics_to_log)
            
def data_generator(inputs, targets, batch_size, weights=None):
    if weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size)
    else:       
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets, weights))
        dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size)
    return dataset

def objective(trial, dataset, model_type, is_weighted_implicit, R_bar_train, R_bar_val):
    def generate_layer_configurations(num_layers, latent_dim, input_feature_len):
        # Define fixed configurations for each combination of num_layers and latent_dim
        configurations = {
            (1, 2): ([2], [input_feature_len]),
            (1, 4): ([4], [input_feature_len]),
            (1, 8): ([8], [input_feature_len]),
            (1, 16): ([16], [input_feature_len]),
            (1, 32): ([32], [input_feature_len]),
            (1, 64): ([64], [input_feature_len]),

            (2, 2): ([20, 2], [20, input_feature_len]),
            (2, 4): ([50, 4], [50, input_feature_len]),
            (2, 8): ([80, 8], [80, input_feature_len]),
            (2, 16): ([150, 16], [150, input_feature_len]),
            (2, 32): ([150, 32], [150, input_feature_len]),

            (3, 2): ([150, 20, 2], [20, 150, input_feature_len]),
            (3, 4): ([150, 40, 4], [150, 40, input_feature_len]),
            (3, 8): ([200, 80, 8], [80, 200, input_feature_len]),
            (3, 16): ([200, 100, 16], [100, 200, input_feature_len]),
            (3, 32): ([200, 100, 32], [100, 200, input_feature_len]),
            # Add more combinations as needed
        }

        return configurations.get((num_layers, latent_dim), ([], []))

    # Shared hyperparameters
    latent_dim = trial.suggest_categorical('latent_dim', [2, 4, 8, 16, 32])
    num_layers = trial.suggest_int('num_layers', 1, 2)

    # Get predefined layer configurations
    input_feature_len = R_bar_train.shape[1] if model_type == "user" else R_bar_train.shape[0]
    encoder_layers, decoder_layers = generate_layer_configurations(num_layers, latent_dim, input_feature_len)

    regularizer_factor = trial.suggest_categorical('regularizer_factor', [0, 1e-3])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.0, 0.2])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

    # Create a unique run name using the trial number and hyperparameters
    run_name = f"dataset_{dataset}, trial_{trial.number}, type_{model_type}, is_weighted_implicit_{is_weighted_implicit}, latent_{latent_dim}, layers_{num_layers}, regularizer_factor_{regularizer_factor}, dropout_rate_{dropout_rate}, use_batch_norm_{use_batch_norm}"
    
    # Initialize a new wandb run for each trial
    wandb.init(project="autoencoder_optimization_3", name=run_name, reinit=True, entity="munchong915",  # Replace with your WandB account name
               config={
                   'dataset': dataset,
                   'model_type': model_type,
                   'is_weighted_implicit': is_weighted_implicit,
                   'latent_dim': latent_dim,
                   'num_layers': num_layers,
                   'encoder_layers': encoder_layers,
                   'decoder_layers': decoder_layers,
                   'regularizer_factor': regularizer_factor,
                   'dropout_rate': dropout_rate,
                   'use_batch_norm': use_batch_norm
               })

    # Compute weighted matrix if is_weighted_implicit is True
    # weights = None
    if is_weighted_implicit:
        R_bar_train = MatrixProcessor.compute_terms(R_bar_train)

    # Initialize the model based on the chosen type and condition
    if model_type == "user":
        if is_weighted_implicit:
            autoencoder = WeightedUserAutoEncoder(R_bar_train.shape[1], latent_dim, encoder_layers, decoder_layers, regularizer_factor, dropout_rate, use_batch_norm)
        else:
            autoencoder = UserAutoEncoder(R_bar_train.shape[1], latent_dim)
        train_data, val_data = R_bar_train, R_bar_val
    elif model_type == "item":
        if is_weighted_implicit:
            autoencoder = WeightedItemAutoEncoder(R_bar_train.shape[0], latent_dim, encoder_layers, decoder_layers, regularizer_factor, dropout_rate, use_batch_norm)
        else:
            autoencoder = ItemAutoEncoder(R_bar_train.shape[0], latent_dim)
        train_data, val_data = tf.transpose(R_bar_train), tf.transpose(R_bar_val)
    else:
        raise ValueError("Invalid model type specified")

    # Compile the model with either the custom loss or a standard loss
    if is_weighted_implicit:
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
        # Create datasets
        train_dataset = data_generator(train_data, train_data, batch_size=64)
        val_dataset = data_generator(train_data, train_data, batch_size=64)
    else:
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        # Create datasets
        train_dataset = data_generator(train_data, train_data, batch_size=64)
        val_dataset = data_generator(val_data, val_data, batch_size=64)

    # Set up EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    autoencoder.fit(train_dataset, 
                    epochs=1000, 
                    # batch_size=128,
                    validation_data=val_dataset, 
                    verbose=2, 
                    callbacks=[WandbLogger(), early_stopping])
    
    if not is_weighted_implicit:
        # Use the trained model to make predictions on R_bar_train
        pred_val = autoencoder.predict(train_dataset)

        k_values = [5, 10, 20, 50, 100]
        avg_recall_at_k, avg_random_recall_at_k = calculate_recall_at_k(train_data, val_data, pred_val, k_values)

        # Log each recall_at_k and random_recall_at_k value as separate columns
        for k in k_values:
            wandb.log({f'recall_at_{k}': avg_recall_at_k[k], f'random_recall_at_{k}': avg_random_recall_at_k[k]})

    wandb.finish()
    
    return -avg_recall_at_k[5]  # Optimize for a specific recall value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder Hyperparameter Optimization')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    #parser.add_argument('--p_value', type=float, required=True, help='P value for data preprocessing')
    parser.add_argument('--model_type', type=str, required=True, choices=['user', 'item'], help='Model type to tune (user/item)')
    # parser.add_argument('--is_weighted_implicit', action='store_true', help='Whether or not to use weighted implicit matrix')
    parser.add_argument("--is_weighted_implicit", type=str, choices=["yes", "no"], required=True, help="Use weighted implicit matrix (yes or no)")
    # parser.add_argument('--is_weighted_implicit', type=bool, required=True, help='Whether or not to use weighted implicit matrix')

    args = parser.parse_args()

    dataset = args.dataset
    #p_value = args.p_value
    model_type = args.model_type
    is_weighted_implicit = args.is_weighted_implicit == "yes"
    
    print('Is weighted implicit :', is_weighted_implicit)
    print(is_weighted_implicit)
    # Define the hyperparameter grid
    search_space = {
        'latent_dim': [2, 4, 8, 16, 32],
        'num_layers': list(range(1, 3)),
        'regularizer_factor': [0, 1e-3],
        'dropout_rate': [0.0, 0.2],
        'use_batch_norm': [True, False]
    }

    # Prepare dataset
    R_bar_train, R_bar_val, _, _, _, _ = preprocess_data(dataset=dataset, p=0.0)

    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(
        lambda trial: objective(trial, dataset, model_type, is_weighted_implicit, R_bar_train, R_bar_val),
        n_trials=None,  # Allow Optuna to run through all combinations in the grid,
        n_jobs=1
    )


import argparse
import wandb
import tensorflow as tf

from models.IGMC import IGMC
from models.IGMC_utils import *
from scipy.sparse import csr_matrix, csc_matrix
from utils.data_utils import prepare_data  # Assuming this function exists in your utils module

def main(args):
    # Extract arguments
    dataset_name = args.dataset
    p_value = args.p_value
    lr = args.lr
    step = args.step
    lambda_ = args.lambda_

    # Prepare data
    print('Preparing dataset')
    R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test = prepare_data(dataset_name, p_value)
    print('Done preparing dataset')

    # Initialize wandb
    wandb.init(project='IGMC_2', 
               name=f'IGMC-dataset={dataset_name}, p={p_value}, Lambda_: {lambda_}',
               entity="munchong915",  # Replace with your WandB account name
               config={
                   'dataset': dataset_name,
                   'p_value': p_value,
                   'lambda': lambda_,
                   'step': step,
                   'lr': lr,
               })

    R_train_csr = csr_matrix(R_train)
    R_train_csc = csc_matrix(R_train)
    Arow_train = SparseRowIndexer(R_train_csr)
    Acol_train = SparseColIndexer(R_train_csc)

    train_rows, train_cols, train_labels = sparse.find(R_train_csr)
    train_indices = np.row_stack((train_rows, train_cols))

    R_val_csr = csr_matrix(R_val)
    R_val_csc = csc_matrix(R_val)
    Arow_val = SparseRowIndexer(R_val_csr)
    Acol_val = SparseColIndexer(R_val_csc)

    val_rows, val_cols, val_labels = sparse.find(R_val_csr)
    val_indices = np.row_stack((val_rows, val_cols))

    R_test_csr = csr_matrix(R_test)
    R_test_csc = csc_matrix(R_test)
    Arow_test = SparseRowIndexer(R_test_csr)
    Acol_test = SparseColIndexer(R_test_csc)

    test_rows, test_cols, test_labels = sparse.find(R_test_csr)
    test_indices = np.row_stack((test_rows, test_cols))

    processed_file_path = f'./processed_files/IGMC_data/p={p_value}'
    train_dataset = MyDataset(Arow_train, Acol_train, train_indices, train_labels, processed_file_path=f'{processed_file_path}/train_set')
    val_dataset = MyDataset(Arow_val, Acol_val, val_indices, val_labels, processed_file_path=f'{processed_file_path}/val_set')
    test_dataset = MyDataset(Arow_test, Acol_test, test_indices, test_labels, processed_file_path=f'{processed_file_path}/test_set')

    # if not os.path.exists(processed_file_path):
    train_subgraphs = train_dataset.process()
    val_subgraphs = val_dataset.process()
    test_subgraphs = test_dataset.process()

    # Train
    BATCH_SIZE = 200
    EPOCHS = step
    learning_rate = lr

    # Create the IGMC model
    h = 1  # This is based on previous discussions; adjust as needed.
    node_dim = 2*h + 2
    layer = 4
    hidden_dim = 32
    ratings = [0, 1, 2, 3, 4]

    igmc = IGMC(node_dim, layer, hidden_dim, ratings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Early stopping parameters
    patience = 50 # Number of epochs to wait for improvement before stopping
    
    # Early stopping setup
    #early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
    best_val_rmse = float('inf')
    patience_counter = 0

    # Training Loop
    for epoch in range(EPOCHS):
        print(f"\nStart of epoch {epoch + 1}/{EPOCHS}")
        # Adjust the learning rate every 20 epochs
        if (epoch + 1) % 50 == 0:
            new_learning_rate = optimizer.learning_rate * 0.1
            optimizer.learning_rate.assign(new_learning_rate)
            print(f"Updated learning rate to: {optimizer.learning_rate.numpy()}")

        # Shuffle the training data for each epoch
        np.random.shuffle(train_subgraphs)
        
        # Iterate over batches
        num_batches = len(train_subgraphs) // BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_subgraphs = train_subgraphs[start_idx:end_idx]
            
            # Extract data for this batch
            processed_batch = pyg_batching(batch_subgraphs)

            # Train on this batch
            with tf.GradientTape() as tape:
                # Forward pass
                H_concat = igmc([processed_batch['x'], processed_batch['edge_index'], processed_batch['edge_type'], processed_batch['batch']], training=True)

                # Get the predicted ratings for all nodes
                predicted_ratings_all = igmc.mlp(H_concat)
                
                total_loss = igmc.compute_loss(processed_batch['y'], predicted_ratings_all)

            # print("Total loss :", total_loss)
            # train_rmse = compute_rmse(batch_subgraphs, igmc, BATCH_SIZE)
            # print("Train rmse :", train_rmse)
            
            # Compute gradients and update model
            grads = tape.gradient(total_loss, igmc.trainable_weights)
            optimizer.apply_gradients(zip(grads, igmc.trainable_weights))       
                        
        # Compute RMSE for train, validation, and test sets at the end of each epoch
        train_rmse = compute_rmse(train_subgraphs, igmc, BATCH_SIZE)
        val_rmse = compute_rmse(val_subgraphs, igmc, BATCH_SIZE)  # Assuming you have a variable named val_dataset
        test_rmse = compute_rmse(test_subgraphs, igmc, BATCH_SIZE)  # Assuming you have a variable named test_dataset

        # Log metrics to wandb with 4 decimal places
        wandb.log({
            "epoch": epoch,
            "Loss/Train": round(train_rmse, 4),
            "Loss/Validation": round(val_rmse, 4),
            "Loss/Test": round(test_rmse, 4),
        })


        # Print string-formatted values in the logs
        print(f"Epoch {epoch}:")  # Assuming you have an epoch variable
        print(f"Loss/Train: {'{:.5f}'.format(float(train_rmse))}")
        print(f"Loss/Validation: {'{:.5f}'.format(float(val_rmse))}")
        print(f"Loss/Test: {'{:.5f}'.format(float(test_rmse))}")
        
        # Early stopping check
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    wandb.finish()
    
    return train_rmse, val_rmse, test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for training")
    parser.add_argument("--p_value", type=float, required=True, help="Percentage of explicit data to be removed")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate of U and V")
    parser.add_argument("--step", type=int, required=True, help="Step size")
    parser.add_argument("--lambda_", type=float, required=True, help="Lambda parameter")
    # Add other arguments as needed

    args = parser.parse_args()
    best_train_rmse, best_val_rmse, best_test_rmse = main(args)

    print(f"Best RMSE - Train: {best_train_rmse}, Validation: {best_val_rmse}, Test: {best_test_rmse}")

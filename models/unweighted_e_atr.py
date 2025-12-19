from autoencoders.autoencoder import UserAutoEncoder, ItemAutoEncoder

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import wandb

import os

def data_generator(inputs, targets, batch_size, weights=None):
    if weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size)
    else:       
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets, weights))
        dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size)
    return dataset

class Unweighted_E_ATR:
    def __init__(self, config, dataset_name):
        # Print the configuration dictionary
        print("Configuration:", config)

        # Set default values or raise errors for missing mandatory parameters
        self.latent_side_info = config.get('latent_side_info', 8)  # Replace 'default_value' with actual default or error handling
        self.latent_M = config.get('latent_M', 8)
        self.lambda_M = config.get('lambda_M', 0.0)
        self.step = config.get('step', 1000)
        self.lr_M = config.get('lr_M', 0.01)
        self.lr_ae = config.get('lr_ae', 0.01)
        self.p = config.get('p_value', 0.0)
        self.embeddings_epochs = config.get('embeddings_epochs', 200)
    
        # Paths for weight files
        self.autoencoder_X_weights_path = (f'ae_weights/{dataset_name}/Unweighted_DAMC/'
                                           f'latent_dim={self.latent_side_info}/autoencoder_X_weights.h5')
        self.autoencoder_Y_weights_path = (f'ae_weights/{dataset_name}/Unweighted_DAMC/'
                                           f'latent_dim={self.latent_side_info}/autoencoder_Y_weights.h5')

        # Check if weights exist
        self.weights_exist = self._check_weights(self.autoencoder_X_weights_path) and \
                             self._check_weights(self.autoencoder_Y_weights_path)


    def _check_weights(self, path):
        """Check if weights file exists at the given path."""
        return os.path.exists(path)

    def compute_probabilities(self, matrix):
        # Sum the entries in each row
        row_sums = tf.reduce_sum(matrix, axis=1)

        # Total sum of entries in the matrix
        total_entries = tf.reduce_sum(matrix)

        # Compute empirical probabilities
        p_hat = row_sums / total_entries

        # Compute smoothed probabilities
        m = matrix.shape[0]  # Number of rows in the matrix
        p_check = 0.5 * (p_hat + 1/m)

        return p_hat, p_check

    @tf.function
    def compute_terms(self, implicit_matrix):
        # Compute the latent representations of the rows and columns
        X = self.autoencoder_X.encode(implicit_matrix)
        Y = self.autoencoder_Y.encode(tf.transpose(implicit_matrix))

        # Compute the products of X, Y, and the probability matrices
        product_X = tf.matmul(tf.matmul(tf.transpose(X), self.diagonal_P), X)
        product_Y = tf.matmul(tf.matmul(tf.transpose(Y), self.diagonal_Q), Y)

        # Compute the eigenvalues and eigenvectors of these products
        eigenvalues_X, P_hat = tf.linalg.eigh(product_X)
        eigenvalues_Y, Q_hat = tf.linalg.eigh(product_Y)

        # Compute D_hat and E_hat
        D_hat = tf.linalg.diag(eigenvalues_X)
        E_hat = tf.linalg.diag(eigenvalues_Y)

        # Compute D_check and E_check
        identity = tf.eye(self.latent_M, dtype=tf.float32)
        D_check = (0.5 * D_hat) + ((1/(2 * self.latent_M)) * identity)
        E_check = (0.5 * E_hat) + ((1/(2 * self.latent_M)) * identity)

        # Compute the square roots and their inverses
        sqrt_D_check = tf.sqrt(D_check)
        sqrt_E_check = tf.sqrt(E_check)
        inv_sqrt_D_check = tf.linalg.inv(sqrt_D_check)
        inv_sqrt_E_check = tf.linalg.inv(sqrt_E_check)

        # Compute the inverses of P_hat and Q_hat
        inv_P_hat = tf.linalg.inv(P_hat)
        inv_Q_hat = tf.linalg.inv(Q_hat)

        # Compute M_check, X_check, and Y_check
        M = tf.matmul(self.U, tf.transpose(self.V))
        M_check = tf.matmul(tf.matmul(tf.matmul(sqrt_D_check, P_hat), M), tf.matmul(inv_Q_hat, sqrt_E_check))
        X_check = tf.matmul(X, tf.matmul(inv_P_hat, inv_sqrt_D_check))
        Y_check = tf.matmul(tf.matmul(inv_sqrt_E_check, Q_hat), tf.transpose(Y))

        # Compute M's nuclear norm
        # Compute the singular values of M using SVD
        singular_values = tf.linalg.svd(M_check, compute_uv=False)
        # Compute the nuclear norm, which is the sum of the singular values
        nuclear_norm = tf.reduce_sum(singular_values)

        XMY_check = tf.matmul(tf.matmul(X_check, M_check), Y_check)

        #Z = tf.matmul(self.Z_1, tf.transpose(self.Z_2))
        #Z_check = tf.matmul(tf.matmul(self.P_check, Z), self.Q_check)

        return XMY_check, nuclear_norm

    def fit(self, R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test):
        tf.random.set_seed(42)

        self.U, self.V = self._init_low_rank_matrices(R_train)
        
        # Prepare data generator for autoencoder training
        batch_size = 32
        user_dataset = tf.data.Dataset.from_tensor_slices((R_bar_train, R_bar_train))
        user_dataset = user_dataset.shuffle(buffer_size=100).batch(batch_size)

        item_dataset = tf.data.Dataset.from_tensor_slices((tf.transpose(R_bar_train), tf.transpose(R_bar_train)))
        item_dataset = item_dataset.shuffle(buffer_size=100).batch(batch_size)

        # Initialize autoencoder
        ae_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_ae)

        self.autoencoder_X = UserAutoEncoder(R_bar_train.shape[1], self.latent_side_info)
        self.autoencoder_X.compile(optimizer=ae_optimizer, loss='binary_crossentropy')

        self.autoencoder_Y = ItemAutoEncoder(R_bar_train.shape[0], self.latent_side_info)
        self.autoencoder_Y.compile(optimizer=ae_optimizer, loss='binary_crossentropy')

        # Dummy forward pass to build the model (ensure weights are created)
        _ = self.autoencoder_X(R_bar_train)
        _ = self.autoencoder_Y(tf.transpose(R_bar_train))

        # Set up EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Check if weights exist for autoencoder_X
        if self._check_weights(self.autoencoder_X_weights_path):
            self.autoencoder_X.load_weights(self.autoencoder_X_weights_path)
        else:
            self.autoencoder_X.fit(user_dataset, validation_data=user_dataset, epochs=200, verbose=2, batch_size=batch_size, callbacks=[early_stopping])

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.autoencoder_X_weights_path), exist_ok=True)

            self.autoencoder_X.save_weights(self.autoencoder_X_weights_path)

        # Check if weights exist for autoencoder_Y
        if self._check_weights(self.autoencoder_Y_weights_path):
            self.autoencoder_Y.load_weights(self.autoencoder_Y_weights_path)
        else:
            self.autoencoder_Y.fit(item_dataset, validation_data=item_dataset, epochs=200, verbose=2, batch_size=batch_size, callbacks=[early_stopping])

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.autoencoder_Y_weights_path), exist_ok=True)

            self.autoencoder_Y.save_weights(self.autoencoder_Y_weights_path)


        # Matrix Factorization
        M_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_M)

        # Define terms
        reg_normalizer = tf.math.sqrt(tf.cast(R_bar_train.shape[0] * R_bar_train.shape[1], tf.float32))

        train_mask = self._create_mask(R_train)
        val_mask = self._create_mask(R_val)
        test_mask = self._create_mask(R_test)

        # Early stopping setup
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
        best_val_rmse = float('inf')
        patience_counter = 0

        # Pre-compute E_ATR terms
        # Compute the empirical and smoothed probabilities
        self.p_hat, self.p_check = self.compute_probabilities(R_bar_train)
        self.q_hat, self.q_check = self.compute_probabilities(tf.transpose(R_bar_train))
    
        # Convert the probabilities to diagonal matrices
        self.diagonal_P = tf.linalg.diag(self.p_hat)
        self.diagonal_Q = tf.linalg.diag(self.q_hat)
        self.P_check = tf.linalg.diag(self.p_check)
        self.Q_check = tf.linalg.diag(self.q_check)

        for i in range(self.step):
            with tf.GradientTape(persistent=True) as tape:
                XMY_check, M_nuclear_norm = self.compute_terms(R_bar_train)
                
                # Compute squared loss
                train_loss = tf.reduce_sum(train_mask * tf.math.squared_difference(R_train, XMY_check)) / tf.reduce_sum(train_mask)
                val_loss = tf.reduce_sum(val_mask * tf.math.squared_difference(R_val, XMY_check)) / tf.reduce_sum(val_mask)
                test_loss = tf.reduce_sum(test_mask * tf.math.squared_difference(R_test, XMY_check)) / tf.reduce_sum(test_mask)
         
                # Compute regularization loss
                #loss_U = self.lambda_M * tf.reduce_sum(tf.math.square(self.U)) / reg_normalizer
                #loss_V = self.lambda_M * tf.reduce_sum(tf.math.square(self.V)) / reg_normalizer
                loss_M_check = (self.lambda_M) * M_nuclear_norm

                loss_autoencoder_X = tf.reduce_mean(tf.keras.losses.binary_crossentropy(R_bar_train, self.autoencoder_X(R_bar_train)))
                loss_autoencoder_Y = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.transpose(R_bar_train), self.autoencoder_Y(tf.transpose(R_bar_train))))

                total_loss = train_loss + loss_M_check + loss_autoencoder_X + loss_autoencoder_Y
     
            # Do backpropagation
            M_grads = tape.gradient(total_loss, [self.U, self.V])
            M_optimizer.apply_gradients(zip(M_grads, [self.U, self.V]))

            autoencoder_variables = self.autoencoder_X.trainable_variables + self.autoencoder_Y.trainable_variables
            ae_grads = tape.gradient(total_loss, autoencoder_variables)
            ae_optimizer.apply_gradients(zip(ae_grads, autoencoder_variables))

            del tape

            # Compute rmse
            train_rmse = tf.math.sqrt(train_loss)
            val_rmse = tf.math.sqrt(val_loss)
            test_rmse = tf.math.sqrt(test_loss)

            # Log metrics to wandb
            wandb.log({
                "Loss/Train": float(train_rmse.numpy()),
                "Loss/Validation": float(val_rmse.numpy()),
                "Loss/Test": float(test_rmse.numpy()),
                "Loss/Autoencoder_X": loss_autoencoder_X,
                "Loss/Autoencoder_Y": loss_autoencoder_Y
            })

            # Print string-formatted values in the logs
            print(f"Epoch {i}:")  # Assuming you have an epoch variable
            print(f"Loss/Train: {'{:.5f}'.format(float(train_rmse.numpy()))}")
            print(f"Loss/Validation: {'{:.5f}'.format(float(val_rmse.numpy()))}")
            print(f"Loss/Test: {'{:.5f}'.format(float(test_rmse.numpy()))}")
            print(f"Loss/Autoencoder_X: {'{:.5f}'.format(loss_autoencoder_X)}")
            print(f"Loss/Autoencoder_Y: {'{:.5f}'.format(loss_autoencoder_Y)}")

            # Early stopping check
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping.patience:
                print("Early stopping triggered")
                break


        # Additional logging or return values as needed...
        return train_rmse, val_rmse, test_rmse

    def _create_mask(self, X):
        mask_tf = tf.where(X == 0, tf.zeros_like(X), 1)

        return mask_tf

    def _init_low_rank_matrices(self, R):
        """
        Initialize low-rank matrices U and V with random values.

        :param R: The input matrix for which low-rank matrices are initialized.
        :return: None
        """
        # Initialize U and V with random values from a normal distribution
        # The shape of U and V depends on the shape of the input matrix R and the predefined max_rank
        U = tf.Variable(tf.random.normal((self.latent_side_info, self.latent_M), dtype=tf.float32))
        V = tf.Variable(tf.random.normal((self.latent_side_info, self.latent_M), dtype=tf.float32))

        #Z_1 = tf.Variable(tf.random.normal((R.shape[0], self.latent_side_info), dtype=tf.float32))
        #Z_2 = tf.Variable(tf.random.normal((R.shape[1], self.latent_side_info), dtype=tf.float32))

        return U, V#, Z_1, Z_2
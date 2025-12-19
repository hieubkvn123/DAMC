import tensorflow as tf
import numpy as np
import os

class SVD_XMY:
    def __init__(self, config):
        # Print the configuration dictionary
        print("Configuration:", config)

        # Set default values or raise errors for missing mandatory parameters
        self.latent_side_info = config.get('latent_side_info', 8)
        self.latent_M = config.get('latent_M', 8)
        self.lambda_ = config.get('lambda_', 0.0)
        self.step = config.get('step', 1000)
        self.lr_M = config.get('lr_M', 0.01)
        # self.alpha = config.get('alpha', 1.0)
        self.patience = 100
        
    def fit(self, R_bar_train, R_train, R_val, R_test):
        tf.random.set_seed(42)

        # Use SVD to compute the initial X and Y matrices from R_bar_train
        U, s, Vt = self._compute_svd(R_bar_train, self.latent_side_info)
        #s_matrix = tf.linalg.diag(s)  # Convert singular values into a diagonal matrix
        self.X_init = U
        self.Y_init = Vt

        # Initialize M
        self.U, self.V = self._init_low_rank_matrices()

        batch_size = 32
        num_epochs = self.step
        lr_M = self.lr_M

        M_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_M)

        # Define masks for training, validation, and test sets
        train_mask = self._create_mask(R_train)
        val_mask = self._create_mask(R_val)
        test_mask = self._create_mask(R_test)

        best_val_rmse = float('inf')  # Initialize best validation RMSE
        patience_counter = 0  # Initialize patience counter

        # Training loop
        for epoch in range(num_epochs):
            with tf.GradientTape(persistent=True) as tape:
                XMY = tf.matmul(tf.matmul(self.X_init, self.U), tf.matmul(tf.transpose(self.V), self.Y_init))

                # Compute squared loss for the training, validation, and test sets
                train_loss = tf.reduce_sum(train_mask * tf.math.squared_difference(R_train, XMY)) / tf.reduce_sum(train_mask)
                val_loss = tf.reduce_sum(val_mask * tf.math.squared_difference(R_val, XMY)) / tf.reduce_sum(val_mask)
                test_loss = tf.reduce_sum(test_mask * tf.math.squared_difference(R_test, XMY)) / tf.reduce_sum(test_mask)

                # Regularization loss for U and V
                loss_U = self.lambda_ * tf.reduce_sum(tf.math.square(self.U))
                loss_V = self.lambda_ * tf.reduce_sum(tf.math.square(self.V))

                # Total loss
                total_loss = train_loss + loss_U + loss_V

            # Gradient update
            M_grads = tape.gradient(total_loss, [self.U, self.V])
            M_optimizer.apply_gradients(zip(M_grads, [self.U, self.V]))

            # Compute rmse
            train_rmse = tf.math.sqrt(train_loss)
            val_rmse = tf.math.sqrt(val_loss)
            test_rmse = tf.math.sqrt(test_loss)

            # Log metrics to wandb with 4 decimal places
            wandb.log({
                "epoch": epoch,
                "Loss/Train": round(train_rmse.numpy(), 4),
                "Loss/Validation": round(val_rmse.numpy(), 4),
                "Loss/Test": round(test_rmse.numpy(), 4),
            })
            # Logging and progress
            # if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch}, Train Loss: {train_rmse.numpy()}, Val Loss: {val_rmse.numpy()}, Test Loss: {test_rmse.numpy()}')

            # Early stopping check
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered")
                break

        return train_loss, val_loss, test_loss

    def _compute_svd(self, matrix, k):
        """Compute SVD and keep only the first k singular values/vectors."""
        s, u, vt = tf.linalg.svd(matrix, full_matrices=False)
        u_k = u[:, :k]
        s_k = s[:k]
        vt_k = vt[:k, :]
        return u_k, s_k, vt_k

    def _create_mask(self, X):
        mask_tf = tf.where(X == 0, tf.zeros_like(X), 1)

        return mask_tf

    def _init_low_rank_matrices(self):
        # Initialize low-rank matrices randomly
        U = tf.Variable(tf.random.normal([self.latent_side_info, self.latent_M]), dtype=tf.float32)
        V = tf.Variable(tf.random.normal([self.latent_side_info, self.latent_M]), dtype=tf.float32)

        #Z_1 = tf.Variable(tf.linalg.qr(tf.random.normal([R.shape[0], self.Z_max_rank]))[0], dtype=tf.float32)
        #Z_2 = tf.Variable(tf.linalg.qr(tf.random.normal([R.shape[1], self.Z_max_rank]))[0], dtype=tf.float32)

        return U, V#, Z_1, Z_2

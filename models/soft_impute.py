import tensorflow as tf
import wandb

# Soft Impute Class
class SoftImpute():
    def __init__(self, max_rank, lambda_, step, lr, p, init_with_svd=False, dtype=tf.float32):
        self.max_rank = max_rank
        self.lambda_ = lambda_
        self.step = step
        self.lr = lr
        self.p = p
        self.init_with_svd = init_with_svd
        self.dtype = dtype

    def fit(self, X_input, R_val_input, R_test_input):
        X = tf.cast(X_input, dtype=self.dtype)
        self.U, self.V = self._init_low_rank_matrices(X)
        
        X_tf, mask_tf = self._create_tf_constants(X)
        R_val, R_bar_val = self._create_tf_constants(R_val_input)
        R_test, R_bar_test = self._create_tf_constants(R_test_input)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # train_rmse_ls = []
        # val_rmse_ls = []
        # test_rmse_ls = []
        # grad_norm_U = []
        # grad_norm_V = []
        # frobenius_norm_Z = []

        non_zero_elements = tf.reduce_sum(mask_tf)
        normalizer = tf.math.sqrt(tf.cast(X_input.shape[0] * X_input.shape[1], tf.float32))

        val_non_zero = tf.reduce_sum(R_bar_val)
        test_non_zero = tf.reduce_sum(R_bar_test)
    
        best_val_rmse = float('inf')
        patience_counter = 0
        patience = 100

        for i in range(self.step):
            with tf.GradientTape() as tape:
                # Compute reconstruction error
                Z = tf.matmul(self.U, tf.transpose(self.V))

                loss_rec = tf.reduce_sum(mask_tf * tf.math.squared_difference(X_tf, Z)) / non_zero_elements

                val_rec = tf.reduce_sum(R_bar_val * tf.math.squared_difference(R_val, Z)) / val_non_zero
                test_rec = tf.reduce_sum(R_bar_test * tf.math.squared_difference(R_test, Z)) / test_non_zero


                # Compute regularization terms
                loss_U = self.lambda_ * tf.reduce_sum(tf.math.square(self.U)) / normalizer
                loss_V = self.lambda_ * tf.reduce_sum(tf.math.square(self.V)) / normalizer

                # Total loss
                loss = loss_rec + loss_U + loss_V
            
            grads = tape.gradient(loss, [self.U, self.V])
            optimizer.apply_gradients(zip(grads, [self.U, self.V]))

            train_rmse = tf.math.sqrt(loss_rec).numpy()
            val_rmse = tf.math.sqrt(val_rec).numpy()
            test_rmse = tf.math.sqrt(test_rec).numpy()

            # Logging to wandb
            wandb.log({
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "grad_norm_U": tf.norm(grads[-2]).numpy(),
                "grad_norm_V": tf.norm(grads[-1]).numpy(),
                "frobenius_norm_Z": tf.norm(Z).numpy()
            })

            # Early stopping check
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {i}")
                break

            # train_rmse_ls.append(tf.math.sqrt(loss_rec))
            # val_rmse_ls.append(tf.math.sqrt(val_rec))
            # test_rmse_ls.append(tf.math.sqrt(test_rec))
            
            # grad_norm_U.append(tf.norm(grads[-2]))
            # grad_norm_V.append(tf.norm(grads[-1]))
            

        # self.Z_opt = tf.matmul(self.U, tf.transpose(self.V))

        # hyperparams_norms_dict = {'U': grad_norm_U, 'V': grad_norm_V, 'frobenius_norm_Z': frobenius_norm_Z
        #                           }

        return train_rmse, val_rmse, test_rmse
    
    def predict(self, X_input):
        X = tf.cast(X_input, dtype=self.dtype)
        X_tf, mask_tf = self._create_tf_constants(X)
        return tf.matmul(self.U, tf.transpose(self.V))

    def _init_low_rank_matrices(self, X):
        if self.init_with_svd:
            # Initialize low-rank matrices with SVD
            S_init, U_init, V_init = tf.linalg.svd(tf.where(tf.math.is_nan(X), tf.zeros_like(X), X))
            
            S_init = tf.linalg.diag(S_init[:self.max_rank])
            U = tf.matmul(U_init[:, :self.max_rank], tf.sqrt(S_init))
            V = tf.transpose(tf.matmul(tf.sqrt(S_init), tf.transpose(V_init[:,:self.max_rank])))

        else:
            # Initialize low-rank matrices randomly
            U = tf.random.normal((X.shape[0], self.max_rank), dtype=self.dtype)
            V = tf.random.normal((X.shape[1], self.max_rank), dtype=self.dtype)

        return tf.Variable(U, dtype=self.dtype), tf.Variable(V, dtype=self.dtype)

    def _create_tf_constants(self, X):
        # Assuming zero entries in X are the ones you want to mask
        X_tf = tf.constant(X, dtype=self.dtype)
        mask_tf = tf.constant(tf.where(X == 0, tf.zeros_like(X), 1), dtype=self.dtype)

        # Calculate and print the percentage of zero entries in R and R_removal
        zero_entries = tf.math.count_nonzero(X_tf == 0)
        total_entries = tf.size(X_tf, out_type=tf.int64)
        percentage_zero = (zero_entries / total_entries) * 100

        zero_mask_entries = tf.math.count_nonzero(mask_tf == 0)
        total_mask_entries = tf.size(mask_tf, out_type=tf.int64)
        percentage_zero_mask = (zero_mask_entries / total_mask_entries) * 100

        # print(f"Percentage of zero entries in R: {percentage_zero.numpy():.2f}%")
        # print(f"Percentage of zero entries in R_removal: {percentage_zero_mask.numpy():.2f}%")

        return X_tf, mask_tf
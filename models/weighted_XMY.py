from autoencoders.weighted_mse_autoencoder import UserAutoEncoder, ItemAutoEncoder
from utils.weights_utils import MatrixProcessor

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

class Weighted_XMY:
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
        self.embeddings_epochs = config.get('embeddings_epochs', 100)
        self.alpha = config.get('alpha')
    
        # Paths for weight files
        self.autoencoder_X_weights_path = (f'ae_weights/{dataset_name}/Weighted_XMY/'
                                           f'latent_dim={self.latent_side_info}/autoencoder_X_weights.h5')
        self.autoencoder_Y_weights_path = (f'ae_weights/{dataset_name}/Weighted_XMY/'
                                           f'latent_dim={self.latent_side_info}/autoencoder_Y_weights.h5')

        # Check if weights exist
        self.weights_exist = self._check_weights(self.autoencoder_X_weights_path) and \
                             self._check_weights(self.autoencoder_Y_weights_path)


    def _check_weights(self, path):
        """Check if weights file exists at the given path."""
        return os.path.exists(path)

    def fit(self, R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test):
        tf.random.set_seed(42)
        
        R_bar_train_transpose = tf.transpose(R_bar_train)
        matrix_processor = MatrixProcessor()
        #X, Y = matrix_processor.compute_svd(R_bar_train, self.latent_side_info)
        R_bar_train = matrix_processor.compute_terms(R_bar_train)

        # Initialize M
        self.U, self.V = self._init_low_rank_matrices()

        matrix_processor = MatrixProcessor()
        user_matrix_multiplier = matrix_processor.compute_terms(R_bar_train)
        
        num_rows = R_bar_train.shape[0]
        batch_size = 32
        # user_dataset = tf.data.Dataset.from_tensor_slices((R_bar_train, R_bar_train, user_matrix_multiplier))
        user_dataset = tf.data.Dataset.from_tensor_slices((R_bar_train, R_bar_train))

        user_dataset = user_dataset.shuffle(buffer_size=num_rows).batch(batch_size)

        ae_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_ae)
        M_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_M)

        #print('Latent side info :', self.latent_side_info)
        self.autoencoder_X = UserAutoEncoder(R_bar_train.shape[1], self.latent_side_info)
        # self.autoencoder_X.compile(optimizer=ae_optimizer)
        
        item_matrix_multiplier = matrix_processor.compute_terms(R_bar_train_transpose)
        item_dataset = tf.data.Dataset.from_tensor_slices((tf.transpose(R_bar_train), tf.transpose(R_bar_train)))
        #item_dataset = tf.data.Dataset.from_tensor_slices((R_bar_train_transpose, R_bar_train_transpose, item_matrix_multiplier))

        item_dataset = item_dataset.shuffle(buffer_size=num_rows).batch(batch_size)

        self.autoencoder_Y = ItemAutoEncoder(R_bar_train.shape[0], self.latent_side_info)
        # self.autoencoder_Y.compile(optimizer=ae_optimizer)

        # # # Initialize autoencoder
        # self.autoencoder_X = UserAutoEncoder(R_bar_train.shape[1], self.latent_side_info)
        # self.autoencoder_Y = ItemAutoEncoder(R_bar_train.shape[0], self.latent_side_info)

        # ae_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # M_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.autoencoder_X.compile(optimizer=ae_optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.autoencoder_Y.compile(optimizer=ae_optimizer, loss=tf.keras.losses.MeanSquaredError())

        # Set up EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        # Dummy forward pass to build the model (ensure weights are created)
        _ = self.autoencoder_X(R_bar_train)
        _ = self.autoencoder_Y(R_bar_train_transpose)

        # Check if weights exist for autoencoder_X
        if self._check_weights(self.autoencoder_X_weights_path):
            # Dummy forward pass to build the model (ensure weights are created)
            _ = self.autoencoder_X(R_bar_train)

            self.autoencoder_X.load_weights(self.autoencoder_X_weights_path)
        else:
            self.autoencoder_X.fit(user_dataset, validation_data=user_dataset, epochs=200, verbose=2, batch_size=batch_size, callbacks=[early_stopping])

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.autoencoder_X_weights_path), exist_ok=True)

            self.autoencoder_X.save_weights(self.autoencoder_X_weights_path)

        # Check if weights exist for autoencoder_Y
        if self._check_weights(self.autoencoder_Y_weights_path):
            _ = self.autoencoder_Y(R_bar_train_transpose)
            self.autoencoder_Y.load_weights(self.autoencoder_Y_weights_path)
        else:
            self.autoencoder_Y.fit(item_dataset, validation_data=item_dataset, epochs=200, verbose=2, batch_size=batch_size, callbacks=[early_stopping])

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.autoencoder_Y_weights_path), exist_ok=True)

            self.autoencoder_Y.save_weights(self.autoencoder_Y_weights_path)


        # # Check if weights exist for autoencoder_X
        # if self._check_weights(self.autoencoder_X_weights_path):
        #     self.autoencoder_X.load_weights(self.autoencoder_X_weights_path)
        # else:
        #     self.autoencoder_X.fit(R_bar_train, R_bar_train, epochs=100, verbose=0)

        #     # Ensure the directory exists
        #     os.makedirs(os.path.dirname(self.autoencoder_X_weights_path), exist_ok=True)

        #     self.autoencoder_X.save_weights(self.autoencoder_X_weights_path)

        # # Check if weights exist for autoencoder_Y
        # if self._check_weights(self.autoencoder_Y_weights_path):
        #     self.autoencoder_Y.load_weights(self.autoencoder_Y_weights_path)
        # else:
        #     self.autoencoder_Y.fit(tf.transpose(R_bar_train), tf.transpose(R_bar_train), epochs=100, verbose=0)

        #     # Ensure the directory exists
        #     os.makedirs(os.path.dirname(self.autoencoder_Y_weights_path), exist_ok=True)

        #     self.autoencoder_Y.save_weights(self.autoencoder_Y_weights_path)

        # Define terms
        reg_normalizer = tf.math.sqrt(tf.cast(R_bar_train.shape[0] * R_bar_train.shape[1], tf.float32))

        train_mask = self._create_mask(R_train)
        val_mask = self._create_mask(R_val)
        test_mask = self._create_mask(R_test)

        # Initialize high minimums for the best RMSEs
        best_train_rmse = float('inf')
        best_val_rmse = float('inf')
        best_test_rmse = float('inf')

        # Early stopping parameters
        patience = 100 # Number of epochs to wait for improvement before stopping

        # Initialize variables
        patience_counter = 0
        train_rmse_window = []
        val_rmse_window = []
        test_rmse_window = []

        for i in range(self.step):
            with tf.GradientTape(persistent=True) as tape:
                # Extract latent vectors X and Y
                X = self.autoencoder_X.encode(R_bar_train, training=True)
                #print('X Shape :', X.shape)
                Y = self.autoencoder_Y.encode(R_bar_train_transpose, training=True)
                #print('Y Shape :', Y.shape)

                # Compute prediction term - XMY
                XMY = tf.matmul(tf.matmul(X, tf.matmul(self.U, tf.transpose(self.V))), tf.transpose(Y))

                # Compute squared loss
                train_loss = tf.reduce_sum(train_mask * tf.math.squared_difference(R_train, XMY)) / tf.reduce_sum(train_mask)
                val_loss = tf.reduce_sum(val_mask * tf.math.squared_difference(R_val, XMY)) / tf.reduce_sum(val_mask)
                test_loss = tf.reduce_sum(test_mask * tf.math.squared_difference(R_test, XMY)) / tf.reduce_sum(test_mask)

                # Compute regularization loss
                loss_U = self.lambda_M * tf.reduce_sum(tf.math.square(self.U)) / reg_normalizer
                loss_V = self.lambda_M * tf.reduce_sum(tf.math.square(self.V)) / reg_normalizer

                # Compute autoencoder loss
                loss_autoencoder_X = tf.reduce_mean(tf.keras.losses.mean_squared_error(user_matrix_multiplier, self.autoencoder_X(user_matrix_multiplier)))
                loss_autoencoder_Y = tf.reduce_mean(tf.keras.losses.mean_squared_error(item_matrix_multiplier, self.autoencoder_Y(item_matrix_multiplier)))

                # loss_autoencoder_X = tf.reduce_mean(tf.keras.losses.binary_crossentropy(R_bar_train, self.autoencoder_X(R_bar_train)))
                # loss_autoencoder_Y = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.transpose(R_bar_train), self.autoencoder_Y(tf.transpose(R_bar_train))))

#                 # For autoencoder_X
#                 x_pred = self.autoencoder_X(R_bar_train, training=True)
#                 loss_autoencoder_X = (self.autoencoder_X.compute_custom_loss(R_bar_train, x_pred, user_matrix_multiplier) / self.alpha)

#                 # For autoencoder_Y
#                 y_pred_transposed = self.autoencoder_Y(R_bar_train_transpose, training=True)
#                 loss_autoencoder_Y = (self.autoencoder_Y.compute_custom_loss(R_bar_train_transpose, y_pred_transposed, item_matrix_multiplier) / self.alpha)

                # Total train loss
                total_loss = train_loss + loss_U + loss_V + loss_autoencoder_X + loss_autoencoder_Y

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

            # If the current val rmse is less than the minimum in val rmse window, reset the patience counter
            if i != 0 and val_rmse < min(val_rmse_window):
                patience_counter = 0
            else:
                #If the val rmse didn't improve, increment patience counter
                patience_counter += 1

            # Stop training if patience exceeded
            if patience_counter > patience:
                break

            # Add current RMSEs to the windows
            train_rmse_window.append(train_rmse)
            val_rmse_window.append(val_rmse)
            test_rmse_window.append(test_rmse)

            # Log metrics to wandb with 4 decimal places
            wandb.log({
                "epoch": i,
                "Loss/Train": round(train_rmse.numpy(), 4),
                "Loss/Validation": round(val_rmse.numpy(), 4),
                "Loss/Test": round(test_rmse.numpy(), 4),
                "Loss/Autoencoder_X": round(loss_autoencoder_X.numpy(), 4),
                "Loss/Autoencoder_Y": round(loss_autoencoder_Y.numpy(), 4)
            })

            # Print string-formatted values in the logs
            print(f"Epoch {i}:")  # Assuming you have an epoch variable
            print(f"Loss/Train: {'{:.5f}'.format(float(train_rmse.numpy()))}")
            print(f"Loss/Validation: {'{:.5f}'.format(float(val_rmse.numpy()))}")
            print(f"Loss/Test: {'{:.5f}'.format(float(test_rmse.numpy()))}")
            print(f"Loss/Autoencoder_X: {'{:.5f}'.format(loss_autoencoder_X.numpy())}")
            print(f"Loss/Autoencoder_Y: {'{:.5f}'.format(loss_autoencoder_Y.numpy())}")

        # After training is done or patience is exceeded, get the best RMSE
        min_val_index = val_rmse_window.index(min(val_rmse_window))

        best_train_rmse = train_rmse_window[min_val_index]
        best_val_rmse = val_rmse_window[min_val_index]
        best_test_rmse = test_rmse_window[min_val_index]

        return best_train_rmse, best_val_rmse, best_test_rmse


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
    


# class Weighted_XMY:
#     def __init__(self, config, dataset_name):
#         # Print the configuration dictionary
#         print("Configuration:", config)

#         # Set default values or raise errors for missing mandatory parameters
#         self.latent_side_info = config.get('latent_side_info', 8)  # Replace 'default_value' with actual default or error handling
#         self.latent_M = config.get('latent_M', 8)
#         self.lambda_M = config.get('lambda_M', 0.0)
#         self.step = config.get('step', 1000)
#         self.lr_M = config.get('lr_M', 0.01)
#         self.lr_ae = config.get('lr_ae', 0.01)
#         self.p = config.get('p_value', 0.0)
#         self.embeddings_epochs = config.get('embeddings_epochs', 100)
    
#         # Paths for weight files
#         self.autoencoder_X_weights_path = (f'ae_weights/{dataset_name}/Weighted_XMY/'
#                                            f'p={self.p}/latent_dim={self.latent_side_info}/autoencoder_X_weights.h5')
#         self.autoencoder_Y_weights_path = (f'ae_weights/{dataset_name}/Weighted_XMY/'
#                                            f'p={self.p}/latent_dim={self.latent_side_info}/autoencoder_Y_weights.h5')

#         # Check if weights exist
#         self.weights_exist = self._check_weights(self.autoencoder_X_weights_path) and \
#                              self._check_weights(self.autoencoder_Y_weights_path)


#     def _check_weights(self, path):
#         """Check if weights file exists at the given path."""
#         return os.path.exists(path)
    
#     def fit(self, R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test):
#         tf.random.set_seed(42)

#         self.U, self.V = self._init_low_rank_matrices(R_train)

#         # Compute weighted implicit matrix
#         matrix_processor = MatrixProcessor()
#         user_matrix_multiplier = matrix_processor.compute_terms(R_bar_train)
#         item_matrix_multiplier = matrix_processor.compute_terms(tf.transpose(R_bar_train))

#         # Prepare data generator for autoencoder training
#         batch_size = 32
#         user_dataset = tf.data.Dataset.from_tensor_slices((R_bar_train, R_bar_train, user_matrix_multiplier))
#         user_dataset = user_dataset.shuffle(buffer_size=100).batch(batch_size)

#         item_dataset = tf.data.Dataset.from_tensor_slices((tf.transpose(R_bar_train), tf.transpose(R_bar_train), item_matrix_multiplier))
#         item_dataset = item_dataset.shuffle(buffer_size=100).batch(batch_size)

#         # Initialize autoencoder
#         ae_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_ae)

#         self.autoencoder_X = CustomUserAutoEncoder(R_bar_train.shape[1], self.latent_side_info)
#         self.autoencoder_X.compile(optimizer=ae_optimizer)

#         self.autoencoder_Y = CustomItemAutoEncoder(R_bar_train.shape[0], self.latent_side_info)
#         self.autoencoder_Y.compile(optimizer=ae_optimizer)

#         # Dummy forward pass to build the model (ensure weights are created)
#         _ = self.autoencoder_X(R_bar_train)
#         _ = self.autoencoder_Y(tf.transpose(R_bar_train))

#         # Set up EarlyStopping callback
#         early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

#         # Check if weights exist for autoencoder_X
#         if self._check_weights(self.autoencoder_X_weights_path):
#             self.autoencoder_X.load_weights(self.autoencoder_X_weights_path)
#         else:
#             self.autoencoder_X.fit(user_dataset, validation_data=user_dataset, epochs=100, verbose=2, batch_size=batch_size, callbacks=[early_stopping])

#             # Ensure the directory exists
#             os.makedirs(os.path.dirname(self.autoencoder_X_weights_path), exist_ok=True)

#             self.autoencoder_X.save_weights(self.autoencoder_X_weights_path)

#         # Check if weights exist for autoencoder_Y
#         if self._check_weights(self.autoencoder_Y_weights_path):
#             self.autoencoder_Y.load_weights(self.autoencoder_Y_weights_path)
#         else:
#             self.autoencoder_Y.fit(item_dataset, validation_data=item_dataset, epochs=100, verbose=2, batch_size=batch_size, callbacks=[early_stopping])

#             # Ensure the directory exists
#             os.makedirs(os.path.dirname(self.autoencoder_Y_weights_path), exist_ok=True)

#             self.autoencoder_Y.save_weights(self.autoencoder_Y_weights_path)


#         # Matrix Factorization
#         M_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_M)

#         # Define terms
#         reg_normalizer = tf.math.sqrt(tf.cast(R_bar_train.shape[0] * R_bar_train.shape[1], tf.float32))

#         train_mask = self._create_mask(R_train)
#         val_mask = self._create_mask(R_val)
#         test_mask = self._create_mask(R_test)

#         # Early stopping setup
#         early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
#         best_val_rmse = float('inf')
#         patience_counter = 0
        
#         for i in range(self.step):
#             with tf.GradientTape(persistent=True) as tape:
#                 X = self.autoencoder_X.encode(R_bar_train)
#                 Y = self.autoencoder_Y.encode(tf.transpose(R_bar_train))

#                 XMY = tf.matmul(tf.matmul(X, tf.matmul(self.U, tf.transpose(self.V))), tf.transpose(Y))

#                 # Compute squared loss
#                 train_loss = tf.reduce_sum(train_mask * tf.math.squared_difference(R_train, XMY)) / tf.reduce_sum(train_mask)
#                 val_loss = tf.reduce_sum(val_mask * tf.math.squared_difference(R_val, XMY)) / tf.reduce_sum(val_mask)
#                 test_loss = tf.reduce_sum(test_mask * tf.math.squared_difference(R_test, XMY)) / tf.reduce_sum(test_mask)

#                 # Compute regularization loss
#                 loss_U = self.lambda_M * tf.reduce_sum(tf.math.square(self.U)) / reg_normalizer
#                 loss_V = self.lambda_M * tf.reduce_sum(tf.math.square(self.V)) / reg_normalizer
                
#                 # For autoencoder_X
#                 x_pred = self.autoencoder_X(R_bar_train)
#                 loss_autoencoder_X = self.autoencoder_X.compute_custom_loss(R_bar_train, x_pred, user_matrix_multiplier) / 1500

#                 # For autoencoder_Y
#                 y_true_transposed = tf.transpose(R_bar_train)
#                 y_pred_transposed = self.autoencoder_Y(tf.transpose(R_bar_train))
#                 loss_autoencoder_Y = self.autoencoder_Y.compute_custom_loss(y_true_transposed, y_pred_transposed, item_matrix_multiplier) / 1500

#                 # loss_autoencoder_X = tf.reduce_mean(tf.keras.losses.mean_squared_error(weighted_R_bar_train, self.autoencoder_X(weighted_R_bar_train)))
#                 # loss_autoencoder_Y = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.transpose(weighted_R_bar_train), self.autoencoder_Y(tf.transpose(weighted_R_bar_train))))

#                 total_loss = train_loss + loss_U + loss_V + loss_autoencoder_X + loss_autoencoder_Y
            
#             # Do backpropagation
#             M_grads = tape.gradient(total_loss, [self.U, self.V])
#             M_optimizer.apply_gradients(zip(M_grads, [self.U, self.V]))

#             autoencoder_variables = self.autoencoder_X.trainable_variables + self.autoencoder_Y.trainable_variables
#             ae_grads = tape.gradient(total_loss, autoencoder_variables)
#             ae_optimizer.apply_gradients(zip(ae_grads, autoencoder_variables))

#             del tape

#             # Compute rmse
#             train_rmse = tf.math.sqrt(train_loss)
#             val_rmse = tf.math.sqrt(val_loss)
#             test_rmse = tf.math.sqrt(test_loss)

#             # Log metrics to wandb with 4 decimal places
#             wandb.log({
#                 "epoch": i,
#                 "Loss/Train": round(train_rmse.numpy(), 4),
#                 "Loss/Validation": round(val_rmse.numpy(), 4),
#                 "Loss/Test": round(test_rmse.numpy(), 4),
#                 "Loss/Autoencoder_X": round(loss_autoencoder_X.numpy(), 4),
#                 "Loss/Autoencoder_Y": round(loss_autoencoder_Y.numpy(), 4)
#             })

#             # Print string-formatted values in the logs
#             print(f"Epoch {i}:")  # Assuming you have an epoch variable
#             print(f"Loss/Train: {'{:.5f}'.format(float(train_rmse.numpy()))}")
#             print(f"Loss/Validation: {'{:.5f}'.format(float(val_rmse.numpy()))}")
#             print(f"Loss/Test: {'{:.5f}'.format(float(test_rmse.numpy()))}")
#             print(f"Loss/Autoencoder_X: {'{:.5f}'.format(loss_autoencoder_X.numpy())}")
#             print(f"Loss/Autoencoder_Y: {'{:.5f}'.format(loss_autoencoder_Y.numpy())}")

#             # Early stopping check
#             if val_rmse < best_val_rmse:
#                 best_val_rmse = val_rmse
#                 patience_counter = 0
#             else:
#                 patience_counter += 1

#             if patience_counter >= early_stopping.patience:
#                 print("Early stopping triggered")
#                 break


#         # Additional logging or return values as needed...
#         return train_rmse, val_rmse, test_rmse

#     def _create_mask(self, X):
#         mask_tf = tf.where(X == 0, tf.zeros_like(X), 1)

#         return mask_tf

#     def _init_low_rank_matrices(self, R):
#         """
#         Initialize low-rank matrices U and V with random values.

#         :param R: The input matrix for which low-rank matrices are initialized.
#         :return: None
#         """
#         # Initialize U and V with random values from a normal distribution
#         # The shape of U and V depends on the shape of the input matrix R and the predefined max_rank
#         U = tf.Variable(tf.random.normal((self.latent_side_info, self.latent_M), dtype=tf.float32))
#         V = tf.Variable(tf.random.normal((self.latent_side_info, self.latent_M), dtype=tf.float32))

#         return U, V
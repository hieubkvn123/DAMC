import wandb
import os

import tensorflow as tf
from autoencoders.autoencoder import UserAutoEncoder, ItemAutoEncoder
from autoencoders.weighted_autoencoder import WeightedUserAutoEncoder, WeightedItemAutoEncoder
from tensorflow.keras.callbacks import EarlyStopping

from utils.weights_utils import MatrixProcessor

class WandbLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Initialize an empty dictionary to hold the metrics to log
            metrics_to_log = {}

            # Check for 'mse' and 'val_mse' in logs
            if 'mse' in logs:
                metrics_to_log['mse'] = logs['mse']
            if 'val_mse' in logs:
                metrics_to_log['val_mse'] = logs['val_mse']

            # Alternatively, check for 'loss' and 'val_loss'
            if 'loss' in logs:
                metrics_to_log['loss'] = logs['loss']
            if 'val_loss' in logs:
                metrics_to_log['val_loss'] = logs['val_loss']

            # Log the metrics using wandb
            wandb.log(metrics_to_log)
            
# Other necessary imports...
# def generate_layer_configurations(num_layers, latent_dim, input_feature_len):
#     # Define fixed configurations for each combination of num_layers and latent_dim
#     configurations = {
#         (1, 2): ([2], [input_feature_len]),
#         (1, 4): ([4], [input_feature_len]),
#         (1, 8): ([8], [input_feature_len]),
#         (1, 16): ([16], [input_feature_len]),
#         (1, 32): ([32], [input_feature_len]),
#         (1, 64): ([64], [input_feature_len]),

#         (2, 2): ([20, 2], [20, input_feature_len]),
#         (2, 4): ([40, 4], [40, input_feature_len]),
#         (2, 8): ([80, 8], [80, input_feature_len]),
#         (2, 16): ([150, 16], [150, input_feature_len]),
#         (2, 32): ([150, 32], [150, input_feature_len]),

#         (3, 2): ([150, 20, 2], [20, 150, input_feature_len]),
#         (3, 4): ([400, 40, 4], [40, 400, input_feature_len]),
#         (3, 8): ([200, 80, 8], [80, 200, input_feature_len]),
#         (3, 16): ([200, 100, 16], [100, 200, input_feature_len]),
#         (3, 32): ([200, 100, 32], [100, 200, input_feature_len]),
#         # Add more combinations as needed
#     }

#     return configurations.get((num_layers, latent_dim), ([], []))


def data_generator(inputs, targets, batch_size, weights=None):
    if weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size)
    else:       
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets, weights))
        dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size)
    return dataset

class JointModel:
    def __init__(self, config, dataset_name, wandb_project_name):
        # Print the configuration dictionary
        print("Configuration:", config)

        # Set default values or raise errors for missing mandatory parameters
        self.latent_dim = config.get('latent_dim', 8)  # Replace 'default_value' with actual default or error handling
        #self.num_layers = config.get('num_layer', 2)  # Assuming num_layers is part of the config

        self.M_max_rank = config.get('M_max_rank', 8)
        self.lambda_M = config.get('lambda_M', 0.0)
        self.step = config.get('step', 1000)
        self.lr_M = config.get('lr_M', 1e-3)
        self.lr_ae = config.get('lr_ae', 1e-3)
        self.p = config.get('p_value', 0.0)
        self.embeddings_epochs = config.get('embeddings_epochs', 100)
        self.dtype = config.get('dtype', tf.float32)

        self.is_weighted_implicit = config.get('is_weighted_implicit', False)
    
        # Paths for weight files
        self.autoencoder_X_weights_path = (f'ae_weights/{dataset_name}/Weighted_XMY/'
                                           f'p={self.p}/latent_dim={self.latent_dim}/{self.is_weighted_implicit}/autoencoder_X_weights.h5')
        self.autoencoder_Y_weights_path = (f'ae_weights/{dataset_name}/Weighted_XMY/'
                                           f'p={self.p}/latent_dim={self.latent_dim}/{self.is_weighted_implicit}/autoencoder_Y_weights.h5')

        # Check if weights exist
        self.weights_exist = self._check_weights(self.autoencoder_X_weights_path) and \
                             self._check_weights(self.autoencoder_Y_weights_path)

        self.wandb_project_name = wandb_project_name

    def _check_weights(self, path):
        """Check if weights file exists at the given path."""
        return os.path.exists(path)

    #self.initialize_training(R_train, R_bar, R_val, R_bar_val, R_test, R_bar_test)

    def initialize_training(self, R_train, R_bar, R_val, R_bar_val, R_test, R_bar_test):
        self.R_train, self.R_train_mask = self._create_tf_constants(R_train)
        
        if self.is_weighted_implicit:
            self.R_bar_train = MatrixProcessor.compute_terms(R_bar)
            self.R_bar_train = tf.constant(self.R_bar_train, dtype=tf.float32)
            self.R_bar_val = MatrixProcessor.compute_terms(R_bar)
            self.R_bar_val = tf.constant(self.R_bar_val, dtype=tf.float32)
        else:
            self.R_bar_train = tf.constant(R_bar, dtype=tf.float32)
            self.R_bar_val = tf.constant(R_bar_val, dtype=tf.float32)

        self.R_val, self.R_val_mask = self._create_tf_constants(R_val)
        #self.R_bar_val = tf.constant(R_bar_val, dtype=tf.float32)

        self.R_test, self.R_test_mask = self._create_tf_constants(R_test)
        #self.R_bar_test = tf.constant(R_bar_test, dtype=tf.float32)

        self.U, self.V = self._init_low_rank_matrices(self.R_train)
        
        # Determine layer configurations
        x_input_feature_len = self.R_bar_train.shape[1]
        y_input_feature_len = self.R_bar_train.shape[0]
        #x_encoder_layers, x_decoder_layers = generate_layer_configurations(self.num_layers, self.latent_dim, x_input_feature_len)
        #y_encoder_layers, y_decoder_layers = generate_layer_configurations(self.num_layers, self.latent_dim, y_input_feature_len)
        
        if self.is_weighted_implicit:
            # Initialize weighted autoencoders with the determined layer configurations
            self.autoencoder_X = WeightedUserAutoEncoder(x_input_feature_len, self.latent_dim, x_encoder_layers, x_decoder_layers)
            self.autoencoder_Y = WeightedItemAutoEncoder(y_input_feature_len, self.latent_dim, y_encoder_layers, y_decoder_layers)
        else:
            # Initialize unweighted autoencoders with the determined layer configurations
            self.autoencoder_X = UserAutoEncoder(x_input_feature_len, self.latent_dim)
            self.autoencoder_Y = ItemAutoEncoder(y_input_feature_len, self.latent_dim)

        # # Initialize autoencoders
        # self.autoencoder_X = UserAutoEncoder(self.R_bar.shape[1], self.latent_dim)
        # self.autoencoder_Y = ItemAutoEncoder(self.R_bar.shape[0], self.latent_dim)
        
        # Ensure the autoencoder models are built by calling them with some input
        _ = self.autoencoder_X(self.R_bar_train)  # Call with a single sample
        print(self.autoencoder_X.encode(self.R_bar_train).shape)
        _ = self.autoencoder_Y(tf.transpose(self.R_bar_train))  # Call with a single sample
        print(self.autoencoder_Y.encode(tf.transpose(self.R_bar_train)).shape)

        # Check if pre-trained weights exist
        if self.weights_exist:
            self.autoencoder_X.load_weights(self.autoencoder_X_weights_path)
            self.autoencoder_Y.load_weights(self.autoencoder_Y_weights_path)
        else:
            # Train autoencoders if weights do not exist
            self.train_autoencoders()

    def train_autoencoders(self):
        # Define the training logic for the autoencoders
        # Compile autoencoders with an optimizer and loss function
        if self.is_weighted_implicit:
            loss_function = tf.keras.losses.MeanSquaredError()
        else:
            #loss_function = 'binary_crossentropy'
            loss_function = 'binary_crossentropy'

        self.autoencoder_X.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_function)
        self.autoencoder_Y.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_function)

        # Prepare datasets for both autoencoders
        train_dataset_X = data_generator(self.R_bar_train, self.R_bar_train, batch_size=128)
        val_dataset_X = data_generator(self.R_bar_val, self.R_bar_val, batch_size=128)

        train_dataset_Y = data_generator(tf.transpose(self.R_bar_train), tf.transpose(self.R_bar_train), batch_size=128)
        val_dataset_Y = data_generator(tf.transpose(self.R_bar_val), tf.transpose(self.R_bar_val), batch_size=128)

        # EarlyStopping callback
        #early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        # Train the user autoencoder
        self.autoencoder_X.fit(train_dataset_X, epochs=self.embeddings_epochs, validation_data=val_dataset_X, verbose=2, callbacks=[])

        # Train the item autoencoder
        self.autoencoder_Y.fit(train_dataset_Y, epochs=self.embeddings_epochs, validation_data=val_dataset_Y, verbose=2, callbacks=[])

        # Check and create directories for saving weights if they don't exist
        x_weights_dir = os.path.dirname(self.autoencoder_X_weights_path)
        y_weights_dir = os.path.dirname(self.autoencoder_Y_weights_path)

        if not os.path.exists(x_weights_dir):
            os.makedirs(x_weights_dir)
            print(f"Created directory: {x_weights_dir}")

        if not os.path.exists(y_weights_dir):
            os.makedirs(y_weights_dir)
            print(f"Created directory: {y_weights_dir}")

        # Save the trained weights
        self.autoencoder_X.save_weights(self.autoencoder_X_weights_path)
        self.autoencoder_Y.save_weights(self.autoencoder_Y_weights_path)

        # # Train the autoencoders using self.R_bar and its transpose
        # # You should define the number of epochs and batch size as per your requirement
        # self.autoencoder_X.fit(self.R_bar, self.R_bar, epochs=self.embeddings_epochs, batch_size=32)
        # self.autoencoder_Y.fit(tf.transpose(self.R_bar), tf.transpose(self.R_bar), epochs=self.embeddings_epochs, batch_size=32)

        # # Save the trained weights
        # self.autoencoder_X.save_weights(self.autoencoder_X_weights_path)
        # self.autoencoder_Y.save_weights(self.autoencoder_Y_weights_path)

    def calculate_loss(self, actual, predicted, mask):
        """
        Calculate the loss between the actual and predicted values, considering only the masked elements.

        :param actual: The actual values.
        :param predicted: The predicted values.
        :param mask: A mask indicating which elements to consider in the loss calculation.
        :return: The calculated loss.
        """
        squared_difference = tf.math.squared_difference(actual, predicted)
        loss = tf.reduce_sum(mask * squared_difference) / tf.reduce_sum(mask)
        return loss
    
    def train_step(self, optimizer_M, ae_optimizer, R_train, R_bar_train, R_train_mask, R_val, R_val_mask, R_test, R_test_mask):
        with tf.GradientTape(persistent=True) as tape:
            X = self.autoencoder_X.encode(R_bar_train, training=True)
            Y = self.autoencoder_Y.encode(tf.transpose(R_bar_train), training=True)

            XM = tf.matmul(X, tf.matmul(self.U, tf.transpose(self.V)))
            XMY = tf.matmul(XM, tf.transpose(Y))

            # Training loss
            loss_rec = self.calculate_loss(R_train, XMY, R_train_mask)
            # print(tf.math.count_nonzero(self.R_bar_removal))

            # Validation and Test loss
            val_loss = self.calculate_loss(R_val, XMY, R_val_mask)
            test_loss = self.calculate_loss(R_test, XMY, R_test_mask)

            # Regularization losses
            loss_U = self.lambda_M * tf.reduce_sum(tf.math.square(self.U)) / self.normalizer
            loss_V = self.lambda_M * tf.reduce_sum(tf.math.square(self.V)) / self.normalizer

            # Autoencoder losses
            if self.is_weighted_implicit:
                # Calculate RMSE for autoencoders when is_weighted_implicit is True
                loss_autoencoder_X = self.calculate_rmse(R_bar_train, self.autoencoder_X(R_bar_train), R_bar_train)
                loss_autoencoder_Y = self.calculate_rmse(tf.transpose(R_bar_train), self.autoencoder_Y(tf.transpose(R_bar_train)), tf.transpose(R_bar_train))
            else:
                loss_autoencoder_X = tf.reduce_mean(tf.keras.losses.binary_crossentropy(R_bar_train, self.autoencoder_X(R_bar_train)))
                loss_autoencoder_Y = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.transpose(R_bar_train), self.autoencoder_Y(tf.transpose(R_bar_train))))

            # Total loss
            total_loss = loss_rec + loss_U + loss_V + loss_autoencoder_X + loss_autoencoder_Y

        # Calculate gradients and update model parameters
        grads_M = tape.gradient(total_loss, [self.U, self.V])
        optimizer_M.apply_gradients(zip(grads_M, [self.U, self.V]))

        ae_grads = tape.gradient(total_loss, self.autoencoder_X.trainable_variables + self.autoencoder_Y.trainable_variables)
        ae_optimizer.apply_gradients(zip(ae_grads, self.autoencoder_X.trainable_variables + self.autoencoder_Y.trainable_variables))

        del tape

        return loss_rec, val_loss, test_loss, loss_autoencoder_X, loss_autoencoder_Y

    def fit(self, R_bar, R_bar_val, R_bar_test, R_train, R_val, R_test):
        tf.random.set_seed(42)
        self.initialize_training(R_train, R_bar, R_val, R_bar_val, R_test, R_bar_test)

        optimizer_M = tf.keras.optimizers.Adam(learning_rate=self.lr_M)
        ae_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_ae)
        self.normalizer = tf.math.sqrt(tf.cast(self.R_bar_train.shape[0] * self.R_bar_train.shape[1], tf.float32))

        # Early stopping setup
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for i in range(self.step):
            train_loss, val_loss, test_loss, ae_loss_X, ae_loss_Y = self.train_step(optimizer_M, ae_optimizer, R_train, R_bar, self.R_train_mask, R_val, self.R_val_mask, R_test, self.R_test_mask)
            print(f'Epoch {i+1} train_loss : {tf.math.sqrt(train_loss)}')
            #print(f'Epoch {i+1} val_loss : {tf.math.sqrt(val_loss)}')
            #print(f'Epoch {i+1} test_loss : {tf.math.sqrt(test_loss)}')

            #print(f'Epoch {i+1} ae_X_loss : {ae_loss_X}')
            #print(f'Epoch {i+1} ae_Y_loss : {ae_loss_Y}')

            # # Log metrics to wandb
            # wandb.log({
            #     "epoch": i,
            #     "Loss/Train": tf.math.sqrt(train_loss),
            #     "Loss/Validation": tf.math.sqrt(val_loss),
            #     "Loss/Test": tf.math.sqrt(test_loss),
            #     "Loss/Autoencoder_X": ae_loss_X,
            #     "Loss/Autoencoder_Y": ae_loss_Y
            # })
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping.patience:
                print("Early stopping triggered")
                break
        # Additional logging or return values as needed...
        return train_loss, val_loss, test_loss

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

    def calculate_rmse(self, actual, predicted, mask):
        """Calculate the Root Mean Squared Error."""
        squared_difference = tf.math.squared_difference(actual, predicted)
        masked_squared_difference = mask * squared_difference
        mse = tf.reduce_sum(masked_squared_difference) / tf.reduce_sum(mask)
        return tf.math.sqrt(mse)

    def _init_low_rank_matrices(self, R):
        """
        Initialize low-rank matrices U and V with random values.

        :param R: The input matrix for which low-rank matrices are initialized.
        :return: None
        """
        # Initialize U and V with random values from a normal distribution
        # The shape of U and V depends on the shape of the input matrix R and the predefined max_rank
        U = tf.Variable(tf.random.normal((self.latent_dim, self.M_max_rank), dtype=self.dtype))
        V = tf.Variable(tf.random.normal((self.latent_dim, self.M_max_rank), dtype=self.dtype))

        return U, V
    
    # def recommend(self, user_id, k=10):
    #     # Recommendation logic...
    #     # ...

    # def get_recall_at_k(self, predictions, ground_truth):
    #     # Evaluation logic...
    #     # ...

    # def get_user_ground_truth(self, user_id):
    #     # Ground truth extraction logic...
    #     # ...

    # # Utility methods
    # def _create_tf_constants(self, X):
    #     # Utility logic...
    #     # ...

    # def _init_low_rank_matrices(self, R):
    #     # Utility logic for initializing matrices...
    #     # ...

    # # Additional utility methods as needed...

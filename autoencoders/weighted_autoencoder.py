import tensorflow as tf
import os
from scipy.linalg import svd

class MatrixProcessor:
    def __init__(self):
        pass

    def compute_probabilities(self, matrix):
        # Sum the entries in each row
        row_sums = tf.reduce_sum(matrix, axis=1)

        # Total sum of entries in the matrix
        total_entries = tf.reduce_sum(matrix)

        # Compute empirical probabilities
        p_hat = row_sums / total_entries

        # Compute smoothed probabilities
        m = matrix.shape[0]  # Number of rows in the matrix
        p_check = 0.5 * (p_hat + (1/m))

        return p_hat, p_check

    def compute_terms(self, matrix):
        # Compute the empirical and smoothed probabilities
        p_hat, p_check = self.compute_probabilities(matrix)
        q_hat, q_check = self.compute_probabilities(tf.transpose(matrix))

        # Convert the probabilities to diagonal matrices
#         diagonal_P = tf.cast(tf.linalg.diag(p_hat), dtype=tf.float32)
#         diagonal_Q = tf.cast(tf.linalg.diag(q_hat), dtype=tf.float32)
#         P_check = tf.cast(tf.linalg.diag(p_check), dtype=tf.float32)
#         Q_check = tf.cast(tf.linalg.diag(q_check), dtype=tf.float32)

        # Compute new matrix
        p_check_matrix = tf.tile(tf.expand_dims(p_check, 1), [1, len(q_check)])
        q_check_matrix = tf.tile(tf.expand_dims(q_check, 0), [len(p_check), 1])
        new_matrix = tf.math.rsqrt(p_check_matrix * q_check_matrix)

        return new_matrix

    def compute_svd(self, matrix, latent_dim):
        # Compute the SVD and save the first k singular vectors/values
        U, s, Vt = svd(matrix, full_matrices=False)

        X = tf.cast(U[:, :latent_dim], dtype=tf.float32)
        Y = tf.cast(Vt[:latent_dim, :].T, dtype=tf.float32)

        return X, Y

def elementwise_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7  # Add a small epsilon to avoid log(0) scenario
    bce = - y_true * tf.math.log(y_pred + epsilon) - (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
    return bce

class UserAutoEncoder(tf.keras.Model):
    def __init__(self, input_feature_len, latent_dim):
        super(UserAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_feature_len = input_feature_len
    
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")  # For validation loss

        # Define encoder layers
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.latent_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
        ])

        # Define decoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.input_feature_len, activation='softmax')
        ])

    def custom_loss(self, y_true, y_pred, current_multiplier):
        bce = elementwise_binary_crossentropy(y_true, y_pred)

        # If current_multiplier is 2D and has the same shape as y_true and y_pred, you can directly multiply.
        # If it's 1D (batch-wise multiplier), then you need to expand its dimensions to match bce's shape.
#         if len(current_multiplier.shape) == 1:
#             current_multiplier = tf.expand_dims(current_multiplier, -1)

        custom_loss = bce * current_multiplier

        return tf.reduce_mean(custom_loss)

    def compile(self, optimizer, **kwargs):
        super(UserAutoEncoder, self).compile(optimizer=optimizer, loss=self.custom_loss, **kwargs)


    def train_step(self, data):
        # Unpack the data
        x, y, current_multiplier = data
        # # Print the runtime shapes
        # tf.print('x :', x)
        # tf.print('y :', y)
        # tf.print('current multiplier :', current_multiplier)
        # print("Runtime shape of x:", tf.shape(x))
        # print("Runtime shape of y:", tf.shape(y))
        # print("Runtime shape of current_multiplier:", tf.shape(current_multiplier))

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value using custom loss function
            loss = self.custom_loss(y, y_pred, current_multiplier)
            ## Compute the loss value
            #loss = self.compiled_loss(y, y_pred, current_multiplier, regularization_losses=self.losses)

        # tf.print('custom loss :', loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update custom loss metric
        self.loss_tracker.update_state(loss)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        results = {}
        results['loss'] = self.loss_tracker.result()  # Add the custom loss to the results

        return results
    

    def test_step(self, data):
        x, y, current_multiplier = data

        y_pred = self(x, training=False)
        loss = self.custom_loss(y, y_pred, current_multiplier)

        # Update validation loss metric
        self.val_loss_tracker.update_state(loss)

        results = {"loss": self.val_loss_tracker.result()}  # Only update the validation loss

        return results
    

    def compute_custom_loss(self, y_true, y_pred, current_multiplier):
        # Your custom loss logic using y_true, y_pred, and current_multiplier
        loss = self.custom_loss(y_true, y_pred, current_multiplier)
        mean_loss = tf.reduce_mean(loss)
        return mean_loss

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

class ItemAutoEncoder(tf.keras.Model):
    def __init__(self, input_feature_len, latent_dim):
        super(ItemAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_feature_len = input_feature_len

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")  # For validation loss

        # Define encoder layers
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.latent_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
        ])

        # Define decoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.input_feature_len, activation='softmax')
        ])

    def custom_loss(self, y_true, y_pred, current_multiplier):
        bce = elementwise_binary_crossentropy(y_true, y_pred)

        # If current_multiplier is 2D and has the same shape as y_true and y_pred, you can directly multiply.
        # If it's 1D (batch-wise multiplier), then you need to expand its dimensions to match bce's shape.
#         if len(current_multiplier.shape) == 1:
#             current_multiplier = tf.expand_dims(current_multiplier, -1)

        custom_loss = bce * current_multiplier

        return tf.reduce_mean(custom_loss)

    def compile(self, optimizer, **kwargs):
        super(ItemAutoEncoder, self).compile(optimizer=optimizer, loss=self.custom_loss, **kwargs)


    def train_step(self, data):
        # Unpack the data
        x, y, current_multiplier = data
        # # Print the runtime shapes
        # tf.print('x :', x)
        # tf.print('y :', y)
        # tf.print('current multiplier :', current_multiplier)
        # print("Runtime shape of x:", tf.shape(x))
        # print("Runtime shape of y:", tf.shape(y))
        # print("Runtime shape of current_multiplier:", tf.shape(current_multiplier))

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value using custom loss function
            loss = self.custom_loss(y, y_pred, current_multiplier)
            ## Compute the loss value
            #loss = self.compiled_loss(y, y_pred, current_multiplier, regularization_losses=self.losses)

        # tf.print('custom loss :', loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update custom loss metric
        self.loss_tracker.update_state(loss)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        results = {}
        results['loss'] = self.loss_tracker.result()  # Add the custom loss to the results

        return results
    

    def test_step(self, data):
        x, y, current_multiplier = data

        y_pred = self(x, training=False)
        loss = self.custom_loss(y, y_pred, current_multiplier)

        # Update validation loss metric
        self.val_loss_tracker.update_state(loss)

        results = {"loss": self.val_loss_tracker.result()}  # Only update the validation loss

        return results
    
    
    def compute_custom_loss(self, y_true, y_pred, current_multiplier):
        # Your custom loss logic using y_true, y_pred, and current_multiplier
        loss = self.custom_loss(y_true, y_pred, current_multiplier)
        mean_loss = tf.reduce_mean(loss)
        return mean_loss

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)
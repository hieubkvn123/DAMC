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

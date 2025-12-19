import tensorflow as tf
import numpy as np

from models.IGMC_utils import apply_edge_dropout

class RGCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, ratings, num_bases=4, input_dim=None, name="RGCNLayer", **kwargs):
        super(RGCNLayer, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.ratings = ratings
        self.num_bases = num_bases

    def build(self, input_shape):
        # If input_dim was not provided in the constructor, use the inferred input shape
        if self.input_dim is None:
            self.input_dim = input_shape[-1]

        # Basis matrices
        self.B = [self.add_weight(name=f"B_{k}",
                                  shape=(self.output_dim, self.input_dim),
                                  initializer="random_normal",
                                  trainable=True) for k in range(self.num_bases)]
        
        # Coefficients for each rating
        self.a_r = {rating: self.add_weight(name=f"a_r_{rating}",
                                            shape=(self.num_bases,),
                                            initializer="random_normal",
                                            trainable=True) for rating in self.ratings}
        
        # The shared weight matrix W_0
        self.W_0 = self.add_weight(name="W_0",
                                   shape=(self.output_dim, self.input_dim),
                                   initializer="random_normal",
                                   trainable=True)

        super(RGCNLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, edge_index, edge_type = inputs

        # First term: W_0 x
        output = tf.transpose(tf.matmul(self.W_0, tf.transpose(x)))

        # Iterate over all relation types
        for rating in self.ratings:
            # Mask to filter edges of the current type
            mask = tf.equal(edge_type, rating)

            # Get source and target node indices for edges of the current type
            src_nodes = tf.boolean_mask(edge_index[0], mask)
            tgt_nodes = tf.boolean_mask(edge_index[1], mask)

            # Check if there are any edges of the current type
            if tf.size(src_nodes) == 0:
                continue

            # Compute W_r as a linear combination of basis matrices
            W_r = sum([self.a_r[rating][k] * self.B[k] for k in range(self.num_bases)])

            # Gather the features of the source nodes
            src_node_features = tf.gather(x, src_nodes)

            # Transform the gathered features using W_r
            transformed_features = tf.matmul(W_r, tf.transpose(src_node_features))

            # Aggregate the transformed features for target nodes
            aggregated_features = tf.math.unsorted_segment_sum(tf.transpose(transformed_features), tgt_nodes, num_segments=tf.shape(x)[0])

            # Add to the output
            output += aggregated_features

        return output

    # ... (rest of the class remains unchanged)


    def compute_ARR(self):
        """Compute the Adjacent Rating Regularization (ARR) for the layer."""
        regularization = 0
        rating_keys = list(self.a_r.keys())

        # Compute W_r for each rating
        W_r_values = {}
        for rating in rating_keys:
            W_r_values[rating] = sum([self.a_r[rating][k] * self.B[k] for k in range(self.num_bases)])

        # Compute regularization based on differences between adjacent W_r values
        for i in range(len(rating_keys) - 1):
            regularization += tf.norm(W_r_values[rating_keys[i+1]] - W_r_values[rating_keys[i]])

        return regularization


class IGMC(tf.keras.Model):
    def __init__(self, input_dim, num_layers, hidden_dim, ratings, mlp_dim=128, lambda_reg=0.001, name="IGMC", **kwargs):
        super(IGMC, self).__init__(name=name, **kwargs)
        
        self.rgcn_layers = [RGCNLayer(output_dim=hidden_dim, ratings=ratings, input_dim=input_dim, name=f"RGCN_Layer_0")]
        for i in range(1, num_layers):  
            self.rgcn_layers.append(RGCNLayer(output_dim=hidden_dim, ratings=ratings, input_dim=hidden_dim, name=f"RGCN_Layer_{i}"))

        # Define the MLP with a name
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation='relu', name="MLP_Dense_0"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, name="MLP_Output")
        ], name="MLP_Layer")

        self.lambda_reg = lambda_reg


    def call(self, inputs, training=False, **kwargs):
        X, edge_index, edge_type, batch = inputs  # Note the added batch input
        X_copy = tf.identity(X)

        # Apply edge dropout during training
        if training:
            subgraph = {
                'edge_index': edge_index,
                'edge_type': edge_type
            }
            subgraph = apply_edge_dropout(subgraph)
            edge_index = subgraph['edge_index']
            edge_type = subgraph['edge_type']

        # To store node representations across layers for the final concatenation
        node_representations = []
            
        for layer in self.rgcn_layers:
            X = layer([X, edge_index, edge_type])
            X = tf.math.tanh(X)
            node_representations.append(X)
            
        # Concatenate node representations from all layers
        X_concatenated = tf.concat(node_representations, axis=1)

        # Separate out users and items based on the features
        users = tf.squeeze(tf.where(tf.equal(X_copy[:, 0], 1)))
        items = tf.squeeze(tf.where(tf.equal(X_copy[:, 1], 1)))
    
        # Gather the representations for users and items
        user_representations = tf.gather(X_concatenated, users)
        item_representations = tf.gather(X_concatenated, items)

        # Concatenate users and items representations
        X_user_item = tf.concat([user_representations, item_representations], axis=1)
        
        # Aggregate node features based on the batch tensor
        #aggregated_X = tf.math.segment_sum(X_concatenated, batch)
            
        return X_user_item

    # def call(self, inputs, **kwargs):
    #     X, edge_index, edge_type = inputs
        
    #     # To store node representations across layers for the final concatenation
    #     node_representations = []
        
    #     for layer in self.rgcn_layers:
    #         X = layer([X, edge_index, edge_type])
    #         node_representations.append(X)
        
    #     # Concatenate node representations from all layers
    #     X_concatenated = tf.concat(node_representations, axis=1)
        
    #     return X_concatenated

    def compute_loss(self, true_ratings, predicted_ratings_all):
        # Get indices of the central nodes
        #unique_values, indices = tf.unique(batch)

        # Get the indices of the central nodes
        #central_node_indices = tf.stack([tf.where(batch == val)[0][0] for val in unique_values])

        # Extract predicted ratings for the central nodes
        #predicted_ratings_central = tf.gather(predicted_ratings_all, central_node_indices)
        #predicted_ratings_central = tf.squeeze(predicted_ratings_central, axis=-1)

        # Compute the squared difference for central nodes
        squared_difference = tf.math.square(predicted_ratings_all - true_ratings)

        # Compute the mean squared error for central nodes
        mse_loss = tf.reduce_mean(squared_difference)

        # Compute the ARR regularization loss
        arr_loss = sum([layer.compute_ARR() for layer in self.rgcn_layers])

        return mse_loss + (self.lambda_reg * arr_loss)
import tensorflow as tf
import numpy as np
import multiprocessing as mp
from scipy import sparse
import random
import os
import pickle
from tqdm import tqdm
import time

def compute_rmse(dataset, model, batch_size=1000):
    """
    Computes RMSE for the entire dataset using the provided model.
    
    Args:
    - dataset (list): List of subgraphs for evaluation.
    - model (IGMC): The trained IGMC model.
    
    Returns:
    - RMSE (float): Root Mean Squared Error for the dataset.
    """
    # Iterate over batches
    num_batches = len(dataset) // batch_size
    
    total_squared_error = 0
    total_samples = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_subgraphs = dataset[start_idx:end_idx]
        
        # Batch the entire dataset
        processed_data = pyg_batching(batch_subgraphs)
        num_samples = processed_data['y'].shape[0]
        total_samples += num_samples

        # Predict ratings for the entire dataset
        H_concat = model([processed_data['x'], processed_data['edge_index'], processed_data['edge_type'], processed_data['batch']])
        predicted_ratings_all = model.mlp(H_concat)
        # print("Predicted ratings shape :", predicted_ratings_all.shape)

        # Extract the predicted ratings for the central nodes
#         unique_values, _ = tf.unique(processed_data['batch'])
#         central_node_indices = tf.stack([tf.where(processed_data['batch'] == val)[0][0] for val in unique_values])
#         predicted_ratings_central = tf.gather(predicted_ratings_all, central_node_indices)
#         predicted_ratings_central = tf.squeeze(predicted_ratings_central, axis=-1)
        
        # total_loss = model.compute_loss(processed_data['y'], predicted_ratings_all)
        # print("Compute rmse total_loss :", total_loss)
        # Compute the squared error for the entire dataset
        # print("Processed_data y shape :", processed_data['y'].shape)
        # print("Predicted ratings shape :", predicted_ratings_all.shape)
        # print("Difference shape :", (predicted_ratings_all - processed_data['y']).shape)
        squared_error = tf.math.square(predicted_ratings_all - processed_data['y'])
#         print("Squared error shape :", squared_error.shape)
#         print("Compute rmse squared error :", tf.reduce_mean(squared_error))
#         print("Compute reduce_sum :", tf.reduce_sum(squared_error))
        
        total_squared_error += tf.reduce_sum(squared_error).numpy()
    
    # print("1 total squared :", total_squared_error)
    # print("total_samples :", total_samples)
    # print("rmse :", np.sqrt(total_squared_error / total_samples))
    rmse = np.sqrt(total_squared_error / total_samples)
    return rmse

def parallel_func(args):
    return construct_tf_graph(*subgraph_extraction_labeling_tf(*args))

def apply_edge_dropout(subgraph, dropout_prob=0.2):

    edge_index_np = subgraph['edge_index'].numpy()
    edge_type_np = subgraph['edge_type'].numpy()

    # Generate a mask for edges to keep based on the dropout probability
    mask = np.random.rand(edge_index_np.shape[1]) > dropout_prob

    # Apply the mask to the edge_index and edge_type arrays
    edge_index_np = edge_index_np[:, mask]
    edge_type_np = edge_type_np[mask]

    subgraph['edge_index'] = tf.convert_to_tensor(edge_index_np, dtype=tf.int64)
    subgraph['edge_type'] = tf.convert_to_tensor(edge_type_np, dtype=tf.int64)

    return subgraph


class MyDataset:
    def __init__(self, Arow, Acol, links, labels, h=1, sample_ratio=1.0, max_nodes_per_hop=None, processed_file_path='./processed_files/data', parallel=False):
        self.Arow = Arow
        self.Acol = Acol
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.processed_file_path = processed_file_path
        self.parallel = parallel

    def links2subgraphs(self):
        
        print('Enclosing subgraph extraction begins...')
        
        # Define the arguments for each function call in the parallel processing
        args = [
            ((i, j), self.Arow, self.Acol, g_label, self.h, self.sample_ratio, self.max_nodes_per_hop)
                for i, j, g_label in zip(self.links[0], self.links[1], self.labels)
        ]
        
        
        if not self.parallel:
            g_list = []
            with tqdm(total=len(self.links[0])) as pbar:
                for i, j, g_label in zip(self.links[0], self.links[1], self.labels):
                    tmp = subgraph_extraction_labeling_tf(
                        (i, j), self.Arow, self.Acol, g_label, self.h, self.sample_ratio, self.max_nodes_per_hop
                    )
                    data = construct_tf_graph(*tmp)

                    g_list.append(data)
                    pbar.update(1)

            return g_list
        # else:
        #     start = time.time()
        #     with mp.Pool(mp.cpu_count()) as pool:
        #         results = pool.map_async(parallel_func, args)

        #         remaining = len(args)
        #         pbar = tqdm(total=remaining)
        #         while not results.ready():
        #             remaining_new = results._number_left
        #             pbar.update(remaining - remaining_new)
        #             remaining = remaining_new
        #             time.sleep(1)

        #         results = results.get()
        #         pool.close()

        #     pbar.close()
        #     end = time.time()
        #     print(f"Time elapsed for subgraph extraction: {end-start}s")

        #     return results
    
    def process(self):
        # Check if the processed file already exists
        if os.path.exists(self.processed_file_path):
            print("Processed file already exists. Loading data...")
            return self.load()

        # If the file does not exist, proceed with extracting subgraphs
        print("Extracting and processing subgraphs...")
        data_list = self.links2subgraphs()

        # Serialize and save the data_list using pickle
        os.makedirs(os.path.dirname(self.processed_file_path), exist_ok=True)
        with open(self.processed_file_path, 'wb') as f:
            pickle.dump(data_list, f)
        
        return data_list

    def load(self):
        with open(self.processed_file_path, 'rb') as f:
            loaded_data_list = pickle.load(f)
        
        return loaded_data_list

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return sparse.csc_matrix((data, indices, indptr), shape=shape)


def one_hot(indices, depth):
    return tf.one_hot(indices, depth)

def construct_tf_graph(u, v, r, node_labels, max_node_label, y):
    u, v = tf.convert_to_tensor(u, dtype=tf.float32), tf.convert_to_tensor(v, dtype=tf.float32)
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    edge_index = tf.stack([tf.concat([u, v], axis=0), tf.concat([v, u], axis=0)], axis=0)
    edge_type = tf.concat([r, r], axis=0)
    x = tf.cast(one_hot(node_labels, max_node_label+1), dtype=tf.float32)
    y = tf.convert_to_tensor([y], dtype=tf.float32)
    
    data = {
        'x': x,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'y': y
    }

    return data

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)

def subgraph_extraction_labeling_tf(ind, Arow, Acol, y, h=1, sample_ratio=1.0, max_nodes_per_hop=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    
    for dist in range(1, h+1):
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
        
    subgraph = Arow[u_nodes][:, v_nodes]
    subgraph[0, 0] = 0
    
    # prepare pyg graph constructor input
    u, v, r = sparse.find(subgraph)  # r is 1, 2... (rating labels + 1)
    v += len(u_nodes)
    r = r - 1  # transform r back to rating label
    num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]
    max_node_label = 2*h + 1
    
    return u, v, r, node_labels, max_node_label, y

def pyg_batching(subgraphs):
    """
    Create a PyG-like batched representation from a list of subgraphs.
    
    Parameters:
    - subgraphs: List of subgraphs
    
    Returns:
    - batched_data: Dictionary containing batched node features, edge indices, edge types, and batch vector
    """
    
    # Lists to store batched data
    batched_x = []
    batched_edge_index = []
    batched_edge_type = []
    batch_vector = []
    batched_y = []
    
    # Variables to keep track of the number of nodes seen so far
    nodes_cumsum = 0
    
    for subgraph in subgraphs:
        # Node features
        x = subgraph['x'].numpy()
        batched_x.append(x)
        
        y = subgraph['y'].numpy()
        # Reshape y to have shape (num_nodes, 1)
        y = np.expand_dims(y, axis=1)
        batched_y.append(y)
        
        # Edge indices (adjusted based on the number of nodes seen so far)
        edge_index = subgraph['edge_index'].numpy() + nodes_cumsum
        batched_edge_index.append(edge_index)
        
        # Edge types
        edge_type = subgraph['edge_type'].numpy()
        batched_edge_type.append(edge_type)
        
        # Batch vector
        batch_vector.extend([len(batched_x) - 1] * x.shape[0])
        
        # Update nodes_cumsum
        nodes_cumsum += x.shape[0]
    
    # Concatenate everything to form the batched data
    batched_data = {
        'x': tf.convert_to_tensor(np.concatenate(batched_x, axis=0), dtype=tf.float32),
        'edge_index': tf.convert_to_tensor(np.concatenate(batched_edge_index, axis=1), dtype=tf.int32),
        'edge_type': tf.convert_to_tensor(np.concatenate(batched_edge_type), dtype=tf.int32),
        'y': tf.convert_to_tensor(np.concatenate(batched_y, axis=0), dtype=tf.float32),
        'batch': tf.convert_to_tensor(batch_vector, dtype=tf.int32)
    }
    
    return batched_data


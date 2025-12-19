import tensorflow as tf
import os

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder

def get_project_root():
    # Adjust this to correctly locate your project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def save_data(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

def _create_tf_constants(X):
    X = tf.Variable(X, dtype=tf.float32)
    mask_tf = tf.constant(tf.where(tf.math.is_nan(X), tf.zeros_like(X), 1), dtype=tf.float32)

    return mask_tf

def preprocess_data(seed=42, dataset='ml_100k', p=0.1):
    np.random.seed(seed)
    project_root = get_project_root()

    # Read in u.data file (user-movie ratings)
    ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    if dataset == 'ml_100k':
        ratings = pd.read_csv(os.path.join(project_root, 'data/ml-100k/u.data'), sep='\t', header=None, names=ratings_cols)
        ratings['unix_timestamp'] = pd.to_datetime(ratings['unix_timestamp'], unit='s')
    elif dataset == 'douban':
        ratings_cols = ['user_id', 'movie_id', 'rating']
        ratings = pd.read_csv(os.path.join(project_root, 'data/douban_movie(u3022m6977)/um.txt'), sep='\t', header=None, names=ratings_cols)
    elif dataset == 'yelp':
        ratings_cols = ['user_id', 'movie_id', 'rating']
        ratings = pd.read_csv(os.path.join(project_root, 'data/yelp_data(u14085b14037)/ub.txt'), sep='\t', header=None, names=ratings_cols)
    
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    # Fit and transform the user and movie ids to get the encoded ids
    ratings['user_id'] = user_encoder.fit_transform(ratings['user_id'])
    ratings['movie_id'] = movie_encoder.fit_transform(ratings['movie_id'])

    # Pivot the ratings data to create a user-item matrix
    pivot_df = ratings.pivot_table(values='rating', index='user_id', columns='movie_id')

    # Create a dummy mask matrix with all ones
    mask = np.ones(pivot_df.shape)

    # Shuffle the indices
    indices = np.arange(pivot_df.shape[0] * pivot_df.shape[1])
    np.random.shuffle(indices)

    # Divide the indices into three sub-sets
    train_size = int(0.9 * len(indices))
    val_size = int(0.05 * len(indices))
    subsets = [indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]]

    # Create the sub-mask matrices
    train_mask = np.zeros_like(mask)
    val_mask = np.zeros_like(mask)
    test_mask = np.zeros_like(mask)

    # Set the corresponding entries in each sub-mask matrix to 1
    train_mask[np.unravel_index(subsets[0], mask.shape)] = 1
    val_mask[np.unravel_index(subsets[1], mask.shape)] = 1
    test_mask[np.unravel_index(subsets[2], mask.shape)] = 1

    # Check that the sub-mask matrices have non-overlapping ones
    assert np.all((train_mask + val_mask + test_mask) <= 1)

    train_mask = train_mask.astype(bool)
    val_mask = val_mask.astype(bool)
    test_mask = test_mask.astype(bool)

    # Apply the masks to create the train, validation, and test sets
    pivot_data_train = pivot_df.where(train_mask, other=np.nan)
    pivot_data_val = pivot_df.where(val_mask, other=np.nan)
    pivot_data_test = pivot_df.where(test_mask, other=np.nan)

    R_bar_train_removal = _create_tf_constants(pivot_data_train)
    R_bar_val_removal = _create_tf_constants(pivot_data_val)
    R_bar_test_removal = _create_tf_constants(pivot_data_test)

    # Identify indices where mask is 1 for each dataset
    train_ones = np.where(train_mask)
    val_ones = np.where(val_mask)
    test_ones = np.where(test_mask)

    if p != 0.0:
        # Calculate number of 1s to remove (p percent)
        remove_train = int(p * len(train_ones[0]))

        # Convert np.ndarrays to lists of tuples
        train_ones_list = list(zip(train_ones[0], train_ones[1]))

        # Randomly select indices to change to NaN
        remove_indices_train = np.random.choice(len(train_ones_list), remove_train, replace=False)
        
        pivot_data_train.values[train_ones[0][remove_indices_train], train_ones[1][remove_indices_train]] = np.nan
        
        # # Change selected indices to np.nan
        # for idx in remove_indices_train:
        #     pivot_data_train.at[train_ones_list[idx][0], train_ones_list[idx][1]] = np.nan

    pivot_data_train = tf.Variable(pivot_data_train, dtype=tf.float32)
    pivot_data_val = tf.Variable(pivot_data_val, dtype=tf.float32)
    pivot_data_test = tf.Variable(pivot_data_test, dtype=tf.float32)

    pivot_data_train = tf.constant(tf.where(tf.math.is_nan(pivot_data_train), tf.zeros_like(pivot_data_train), pivot_data_train), dtype=tf.float32)
    pivot_data_val = tf.constant(tf.where(tf.math.is_nan(pivot_data_val), tf.zeros_like(pivot_data_val), pivot_data_val), dtype=tf.float32)
    pivot_data_test = tf.constant(tf.where(tf.math.is_nan(pivot_data_test), tf.zeros_like(pivot_data_test), pivot_data_test), dtype=tf.float32)

    return R_bar_train_removal, R_bar_val_removal, R_bar_test_removal, pivot_data_train, pivot_data_val, pivot_data_test

def prepare_data(dataset, p):
    project_root = get_project_root()

    # Construct the file path relative to the project root
    filename = os.path.join(project_root, f'data/real_datasets/{dataset}/p={p}.pickle')

    if os.path.exists(filename):
        # print(f"Loading preprocessed data for {dataset} from disk...")
        R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test = load_data(filename)
    else:
        # print('Dataset :', dataset)
        # print('p :', p)
        R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test = preprocess_data(dataset=dataset, p=float(p))
        save_data((R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test), filename)

    return R_bar_train, R_bar_val, R_bar_test, R_train, R_val, R_test
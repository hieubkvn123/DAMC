import tensorflow as tf
import numpy as np

def calculate_recall_at_k(R_bar_train, R_bar_val, pred_R_bar_val, k_values):
    num_users = R_bar_train.shape[0]
    recall_at_k_results = {k: [] for k in k_values}
    random_recall_at_k_results = {k: [] for k in k_values}

    for i in range(num_users):
        relevant_items = tf.where(R_bar_val[i] == 1)[:, 0]
        if tf.size(relevant_items) == 0:
            continue

        # Ensure no leakage from training data
        check_train_zeros = tf.gather(R_bar_train[i], relevant_items)
        if tf.reduce_sum(check_train_zeros) > 0:
            continue  # Skip this user if there's leakage

        # Calculate recall for different k values
        for k in k_values:
            recall_at_k, random_recall_at_k = calculate_recall_and_random_for_user(i, k, R_bar_train, R_bar_val, pred_R_bar_val)
            recall_at_k_results[k].append(recall_at_k)
            random_recall_at_k_results[k].append(random_recall_at_k)

    # Calculate average recall@k for actual and random recommendations
    avg_recall_at_k = {k: np.nanmean(recall_at_k_results[k]) for k in k_values}
    avg_random_recall_at_k = {k: np.nanmean(random_recall_at_k_results[k]) for k in k_values}

    return avg_recall_at_k, avg_random_recall_at_k

def calculate_recall_and_random_for_user(i, k, R_bar_train, R_bar_val, pred_R_bar_val):
    relevant_items = tf.where(R_bar_val[i] == 1)[:, 0]
    available_items_indices = tf.where(R_bar_train[i] == 0)[:, 0]
    num_available_items = tf.size(available_items_indices, out_type=tf.int32)

    # Generate random recommendations
    random_indices = tf.random.shuffle(tf.range(num_available_items))[:k]
    random_recommendations = tf.gather(available_items_indices, random_indices)

    # Calculate recall@k for random recommendations
    random_recall_at_k = calculate_recall(relevant_items, random_recommendations)

    # Calculate recall@k for actual recommendations
    top_k_recommendations = tf.argsort(pred_R_bar_val[i], direction='DESCENDING')[:k]
    recall_at_k = calculate_recall(relevant_items, top_k_recommendations)

    return recall_at_k, random_recall_at_k

def calculate_recall(relevant_items, recommendations):
    relevant_items = tf.cast(relevant_items, recommendations.dtype)
    equal_tensor = tf.equal(recommendations[:, None], relevant_items)
    relevant_in_top_k = tf.reduce_any(equal_tensor, axis=1)
    num_relevant_in_top_k = tf.reduce_sum(tf.cast(relevant_in_top_k, tf.int32))
    recall_at_k = num_relevant_in_top_k / tf.size(relevant_items, out_type=tf.int32)
    return recall_at_k.numpy()

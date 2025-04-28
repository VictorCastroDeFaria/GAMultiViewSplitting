import numpy as np

from sklearn.neighbors import NearestNeighbors

# =============================================================================
# 
# =============================================================================

def hasAllViews(solution_dict:dict):
    return not np.any([item == [] for item in solution_dict.values()])


# Only for binary encodings
def checkFeasibility(solution:list):
    # Calculate the sum of elements along axis 1 (rows)
    row_sums = np.sum(solution, axis = 1)

    # Calculate the sum of elements along axis 0 (columns)
    col_sums = np.sum(solution, axis = 0)
   
    # Check Rows
    row_with_zeros = np.any(row_sums == 0)
   
    # Check Columns
    col_with_zeros = np.any(col_sums == 0)

    return not row_with_zeros and not col_with_zeros

# =============================================================================
# 
# =============================================================================

def kdn_score(X, y, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', metric='euclidean').fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbors = indices[:, 1:]
    diff_class = np.tile(y, (k, 1)).transpose() != y[neighbors]
    score = np.sum(diff_class, axis=1) / k
    return score, neighbors

def querySamplekDN(X, y, k, query_sample):
    # Fit the Nearest Neighbors model to training data
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', metric='euclidean').fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbors = indices[:, 1:]
    diff_class = np.tile(y, (k, 1)).transpose() != y[neighbors]

    # Obtaining the kDN for each training data instance
    score = np.sum(diff_class, axis=1) / k

    # Obtaining the nearest neighbors (in training data) of each test data instance
    _, query_nbrs_indices = nbrs.kneighbors(query_sample)

    # Estimates the test instance kDN as the average of its neighbors' kDN values
    estimate_query_kDN = np.mean([[score[_] for _ in neighbor_list] for neighbor_list in query_nbrs_indices], axis = 1)

    return estimate_query_kDN


    


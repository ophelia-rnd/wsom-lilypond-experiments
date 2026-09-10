import numpy as np

from minisom import MiniSom
from experiments.utils.som_knn import get_minimal_path_knn_inds, get_minimal_path_altered_knn_inds

def get_ambient_space_ranking_inds(D, d, d_idx):
    ranked_inds = np.argsort(np.linalg.norm(D - d, axis=1))
    is_x = (ranked_inds == d_idx)
    ranked_inds = ranked_inds[~is_x]
    assert len(ranked_inds) == len(D)-1
    return ranked_inds

def calc_som_trustworthiness_umatrix(som:MiniSom, X, k):
    return calc_som_trustworthiness(som, X, k, knn_method=get_minimal_path_knn_inds)

def calc_som_trustworthiness_umatrix_altered(som:MiniSom, X, k):
    return calc_som_trustworthiness(som, X, k, knn_method=get_minimal_path_altered_knn_inds)

def calc_som_trustworthiness(som:MiniSom, X, k, knn_method):

    N = len(X)
    som_weights = som.get_weights()
    som_shape = som_weights.shape[:2]
    som_winmap = som.win_map(X, return_indices=True)
    som_distmap = som.distance_map(scaling="mean")

    x_penalties_list = []

    for i, x in enumerate(X):

        Xx_ambient_ordering_inds = get_ambient_space_ranking_inds(X, x, i)
        assert i not in Xx_ambient_ordering_inds

        fallout_inds = Xx_ambient_ordering_inds[k:]
        fallout_penalties = np.arange(1, len(fallout_inds)+1)
        
        x_embedding_knn_inds = knn_method(som, som_weights, som_winmap, som_distmap, som_shape, X, x, i, k)
        assert len(x_embedding_knn_inds) == k
        assert i not in x_embedding_knn_inds

        x_penalties = 0

        for j in x_embedding_knn_inds:
            fallout_match = np.where(fallout_inds == j)[0]
            
            if len(fallout_match) > 0:
                x_penalties += fallout_penalties[fallout_match[0]]

        x_penalties_list.append(x_penalties)
    
    trustworthiness = 1 - ((2 / (N*k*(2*N-3*k-1))) * sum(x_penalties_list))

    return trustworthiness

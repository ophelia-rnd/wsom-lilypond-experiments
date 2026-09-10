import numpy as np

from minisom import MiniSom
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def get_som_node_flat_index(x, y, som_shape):
	return np.ravel_multi_index((x, y), som_shape)

def build_grid_adjacency_graph(som_distmap, som_shape, D1, D2):
	
	row, col, weight = [], [], []
	directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

	for d_1 in range(D1):
		for d_2 in range(D2):
			current_idx = get_som_node_flat_index(d_1, d_2, som_shape)
			
			for dx, dy in directions:
				nx, ny = d_1 + dx, d_2 + dy

				if 0 <= nx < D1 and 0 <= ny < D2:
					neighbor_idx = get_som_node_flat_index(nx, ny, som_shape)
					w = max(som_distmap[d_1, d_2], som_distmap[nx, ny])
					
					row.append(current_idx)
					col.append(neighbor_idx)
					weight.append(w)
	
	return row, col, weight

def get_minimal_path_knn_inds(som:MiniSom, som_weights, som_winmap, som_distmap, som_shape, X, x, x_idx, k):

	D1, D2 = som_shape
	num_nodes = D1 * D2

	x_bmu = som.winner(x)
	x_bmu_flat = get_som_node_flat_index(x_bmu[0], x_bmu[1], som_shape)

	row, col, weight = build_grid_adjacency_graph(som_distmap, som_shape, D1, D2)
	graph = csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
	distances = shortest_path(csgraph=graph, directed=False, indices=x_bmu_flat, method='D')
	rankings = list(np.ndenumerate(distances.reshape(som_shape)))
	rankings.sort(key=lambda item: item[1])

	x_related_inds_collection = [(coords, som_winmap.get(coords, [])) for (coords, _) in rankings]
	x_related_inds = np.array([
		idx for (coords, won_inds) in x_related_inds_collection if won_inds
		for idx in np.array(won_inds)[np.argsort(np.linalg.norm(X[won_inds] - som_weights[coords], axis=1))]
	]) # intra-node rearrangement

	is_x = (x_related_inds == x_idx)
	knn_inds = x_related_inds[~is_x][:k]

	return knn_inds

def get_minimal_path_altered_knn_inds(som:MiniSom, som_weights, som_winmap, som_distmap, som_shape, X, x, x_idx, k):

	D1, D2 = som_shape
	num_nodes = D1 * D2

	b2mu_inds = np.argsort(som._distance_from_weights([x]))[0, :2]
	x_bmu_1, x_bmu_2 = zip(*np.unravel_index(b2mu_inds, som_shape))
	x_bmu_1_flat, x_bmu_2_flat = b2mu_inds[0], b2mu_inds[1]

	row, col, weight = build_grid_adjacency_graph(som_distmap, som_shape, D1, D2)
	graph = csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
	distances = shortest_path(csgraph=graph, directed=False, indices=x_bmu_1_flat, method='D')
	rankings = list(np.ndenumerate(distances.reshape(som_shape)))
	rankings.sort(key=lambda item: item[1])

	# alter by moving 2nd BMU to the 1st index

	ranked_coords = [coords for (coords, _) in rankings]
	del rankings

	x_bmu_2_ranked_idx = ranked_coords.index(x_bmu_2)
	ranked_coords.pop(x_bmu_2_ranked_idx)
	ranked_coords.insert(1, x_bmu_2)

	assert ranked_coords[0] == x_bmu_1
	assert ranked_coords[1] == x_bmu_2

	# end of altering

	x_related_inds_collection = [(coords, som_winmap.get(coords, [])) for coords in ranked_coords]
	x_related_inds = np.array([
		idx for (coords, won_inds) in x_related_inds_collection if won_inds
		for idx in np.array(won_inds)[np.argsort(np.linalg.norm(X[won_inds] - som_weights[coords], axis=1))]
	]) # intra-node rearrangement

	is_x = (x_related_inds == x_idx)
	knn_inds = x_related_inds[~is_x][:k]

	return knn_inds

import networkx as nx
import random
import numpy as np


# generate connected random graph with equal sized clusters
def generateRandomGraph(clusterSize, clusterNums, pIntra, pInter):
	while True:
		clusters = [clusterSize] * clusterNums
		G = nx.random_partition_graph(clusters,pIntra,pInter)
		if nx.is_connected(G):
			break
	partition=G.graph['partition']
	idx = 0
	for cluster in partition:
		cluster = list(cluster)
		for node in cluster:
			G.node[node]['x'] = (idx+1)
		idx += 1
	return G

# generate connected random graph with sizes following the geometric distribution
def generateRandomGraphGeometric(clusterNums, minSize = 5, pIntra = 0.7, pInter = 0.01, g = 0.08):
	while True:
		clusters = (np.random.geometric(g, size = clusterNums) + minSize).tolist()
		G = nx.random_partition_graph(clusters,pIntra,pInter)
		if nx.is_connected(G):
			break
	partition=G.graph['partition']
	idx = 0
	for cluster in partition:
		cluster = list(cluster)
		for node in cluster:
			G.node[node]['x'] = (idx+1)
		idx += 1
	
	return G

# some static graph for benchmarks
def generateGraph16Test():
	G = nx.Graph()
	G.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),
					  (4,5),(4,6),(4,7),(5,6),(5,7),(6,7),
					  (8,9),(8,10),(8,11),(9,10),(9,11),(10,11),
					  (12,13),(12,14),(12,15),(13,14),(13,15),(14,15),
					   (2,7),(1,4),
					   (6,11),(5,8),
					   (10,15),(9,12)])
	
	G.node[0]['x'] = 1
	G.node[1]['x'] = 1
	G.node[2]['x'] = 1
	G.node[3]['x'] = 1

	G.node[4]['x'] = 2
	G.node[5]['x'] = 2
	G.node[6]['x'] = 2
	G.node[7]['x'] = 2

	G.node[8]['x'] = 3
	G.node[9]['x'] = 3
	G.node[10]['x'] = 3
	G.node[11]['x'] = 3

	G.node[12]['x'] = 4
	G.node[13]['x'] = 4
	G.node[14]['x'] = 4
	G.node[15]['x'] = 4
	
	return G

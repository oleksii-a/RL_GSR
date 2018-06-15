import numpy as np
import networkx as nx
import random
import collections
import Utils


# random sampling
def simulateRandomSampling(G, numSamples, simulationsCount):
	
	A = nx.to_scipy_sparse_matrix(G)
	D = Utils.adj2inc(A)
	nodesCount = G.number_of_nodes()
	xOriginal = np.zeros(nodesCount)
	for i in range(0,nodesCount):
		xOriginal[i] = G.node[i]['x']

	mseList  = []
	for iter in range(0,simulationsCount):
		indices = random.sample(range(0, nodesCount), numSamples)
		(obj, xRecovered) = Utils.recoverGraphSignal(D, indices, xOriginal[indices])
		mse = np.square(xRecovered - xOriginal).mean()
		mseList.append(mse)
	average = np.mean(mseList)
	minMSE = np.min(mseList)
	return minMSE, average


def gradientBandit(graph,samplesCount, numEpisodes):

	#some constants
	nActions = 7
	alpha = 0.03
	decayRate = 0.99
	batchSize = 10

	# initializations
	W = np.ones(shape=(nActions,1)).flatten()
	rmspropСache = np.zeros(shape=(nActions,1)).flatten()

	allPathesLength = dict(nx.all_pairs_shortest_path_length(graph))
	# create incidence matrix which will be further used in sparse label propagation for recovery
	D = Utils.adj2inc(nx.to_scipy_sparse_matrix(graph))

	#collect original signal into array
	nodesCount = graph.number_of_nodes()
	xOriginal = np.array([graph.node[i]['x'] for i in range(0,nodesCount)])

	# buffer for statistics to keep last 1000 vals
	statMSEBuffer = collections.deque(maxlen=1000) 

	actions = []
	rewards = []
	for episode  in range(0,numEpisodes):
		distribution = (np.exp(W)/np.sum(np.exp(W))).flatten()
		samplingSet = [random.randint(0, nodesCount-1)]
		for timeidx in range(0,samplesCount-1):
		
			currentNode = samplingSet[-1]
			actionIdx = np.random.choice(np.arange(0, len(distribution)), p=distribution)
			actions.append(actionIdx)
			numberOfHops = actionIdx+1
			
			# find all nodes at the distance numberOfHops from the current node
			candidates = []
			for key,value in allPathesLength[currentNode].items():
				if (value == numberOfHops) and (key not in samplingSet):
					candidates.append(key)
			
			# if there are multiple candidate nodes, select randomly among them
			if candidates:				
				samplingSet.append(random.choice(candidates)) 
		
		# recover signal
		objective, xRecovered = Utils.recoverGraphSignal(D, samplingSet, xOriginal[samplingSet])
		mseError = np.square(xRecovered - xOriginal).mean()
		
		r = -mseError

		# assign this reward to all actions which contributed to reconstruction
		rerwardVector = [r for i in range(0,samplesCount-1)]
		rewards = rewards + rerwardVector

		if (episode % batchSize == 0):

			# accumulate gradient over the batch
			gradBuf = np.zeros(shape=(nActions,1)).flatten()
			for i in range(0,len(actions)):
				idx = actions[i]
				gradBuf[idx] -= rewards[i]*(1-distribution[idx])
				indices = np.delete(np.arange(0,nActions),idx)
				gradBuf[indices] += rewards[i]*distribution[indices]
			
			# normalize gradient
			gradBuf = gradBuf / len(actions)

			# update weights using RMSProp technique
			rmspropСache = decayRate*rmspropСache + (1-decayRate)*gradBuf**2
			W -= alpha * gradBuf / (np.sqrt(rmspropСache) + 1e-5)

			actions = []
			rewards = []

		statMSEBuffer.append(mseError)

	distribution = (np.exp(W)/np.sum(np.exp(W))).flatten()
	return distribution, np.mean(statMSEBuffer)


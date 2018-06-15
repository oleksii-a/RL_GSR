import numpy as np
import scipy.sparse as s
from cvxpy import *
import cvxpy

def adj2inc(A):
	M = A.copy()
	M = s.triu(M)
	arr = np.nonzero(M)
	
	edgesCount = len(arr[0])
	nodesCount = M.shape[0]
	edgesIdx = list(range(0, edgesCount))
	vOnes = np.ones(edgesCount).tolist();
	vOnesMinus = (-1*np.ones(edgesCount)).tolist()
	I = s.coo_matrix((vOnesMinus+vOnes, (edgesIdx+edgesIdx,arr[0].tolist()+arr[1].tolist())), shape = (edgesCount, nodesCount))
	return I


def recoverGraphSignal(D, indices, y):
	count = D.shape[1]

	# construct and solve convex optimization problem
	x = Variable(count)
	objective = Minimize(cvxpy.norm(D*x,1))
	constraints = [x[indices] == y]
	prob = Problem(objective,constraints)
	optVal = prob.solve()

	xRecovered = np.asarray(x.value).reshape(1,-1)
	return (objective.value, xRecovered)

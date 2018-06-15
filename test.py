from GraphGeneration import *
from GradientMAB import *

# some test code
graph =  generateRandomGraph(5, 3, 0.7, 0.05)

minMSE, mse1 = simulateRandomSampling(graph, 6, 500)
distribution, mse2 = gradientBandit(graph,6,6000)

print(mse1)
print(mse2)

#importing the necessary libraries
import pennylane as qml
from pennylane import numpy as np

#defining the quantum device
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# Note that we want to convert the state from |0> to |1>
# We have that <sigma | Z | sigma> = -1 when sigma is |1>
# and 
# We have that <sigma | Z | sigma> = 1 when sigma is |1>

# Hence, when the cost function has converged to 0,
# We can say that the we have the optimized parameters for
# Our required transformation

def cost(z):
    return circuit(z)+1

initparams = np.array([0.12, 0.13])

# initialising the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.3)
steps = 150
params = initparams

for i in range(steps):
    params = opt.step(cost, params)
    print("The parameters are:")
    print(params)
    print("Cost function is:")
    print(cost(params))   
print("The final parameters: {}".format(params))

import numpy as np
from hawkes.models import *

def generate_toydata(data):
    out = np.zeros((25, 3, 1))
    for d in data:
        out[d[1], d[0], 0] = 1
    return out

def generate_toyreact(data):
    out = np.zeros((3, 3, 25))
    for d in data:
        out[d[0], d[1], d[2]] = 1
    return out

if __name__ == "__main__":
    # test plain sequence data
    sample_data = [(0,0),(0,15),(1,2),(2,4),(1,18),(2,20)]
    data = generate_toydata(sample_data)
    model = LinearDiscreteHawkes(K=3, dt=2, dt_max=3, reg="L1")
    model.set_data(data)
    model.fit_bfgs()
    print "W: ", model.W
    print "bias: ", model.bias
    #print "influence for 0: ", model.Lambda[:, :, 1]
    # test data with features
    sample_react = [(0, 1, 2), (0, 1, 18)]
    data_react = generate_toyreact(sample_react)
    model1 = NeighborLinDiscreteHawkes(K=3, dt=2, dt_max=3, reg="L1")
    model1.set_data(data, data_react)
    model1.fit_bfgs()
    #print model1.W[0:3, :] + model1.W[3:, :], model1.bias
    print model1.Lambda[:,:,2]
    #print model1.W

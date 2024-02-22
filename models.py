import systems
import numpy as np
import scipy

def get_simple_test():
    # DC_motor
    R= 2.0 # Ohms
    L= 0.5 # Henrys
    Km = .015 # torque constant
    Kb = .015 # emf constant
    Kf = 0.2 # Nms
    J= 0.02 # kg.m^2

    A_mat =  np.matrix([[-R/L, -Kb/L],[Km/J, -Kf/J]])

    init_mean = np.zeros(2)
    init_cov = np.eye(2)
    init_dist = scipy.stats.multivariate_normal(init_mean, init_cov).rvs

    return systems.uncontrolled_LTI(A_mat, init_dist)

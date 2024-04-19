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

    init_mean = np.zeros(2)#5*np.ones(2)
    init_cov = 5*np.eye(2)
    init_dist = scipy.stats.multivariate_normal(init_mean, init_cov).rvs
    traj_length = 1 

    return systems.uncontrolled_LTI(A_mat, init_dist, traj_length)

def nasty_test():
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

    def condition(time, state):
        eps = 1e-5
        return  (np.linalg.norm(state) <= 0+eps) and (np.linalg.norm(state) >= 0-eps)

    # Notes: this test results in a marginally stable system, no matter how large
    # the discontinuity around the origin is it can never fling you away
    # hence you just get stuck at a distance of eps from the origin.
    
    # This probably means that knowldge of the Lipschitz constant is needed theoretically
    # but not in practice...

    # The case for discrete time systems is a little different since then you can get proper kicks

    return systems.uncontrolled_LTI_with_instability(A_mat, init_dist, condition)

def half_unstable_test():
    # DC_motor
    R= 2.0 # Ohms
    L= 0.5 # Henrys
    Km = .015 # torque constant
    Kb = .015 # emf constant
    Kf = 0.2 # Nms
    J= 0.02 # kg.m^2

    A_mat =  np.matrix([[-R/L, -Kb/L],[Km/J, -Kf/J]])

    init_mean = np.matrix([[0],[1]])
    init_cov = np.eye(2)
    init_dist = scipy.stats.multivariate_normal(init_mean, init_cov).rvs

    def condition(time, state):
        return  (state[0] > 0)

    # Notes: this test results in a marginally stable system, no matter how large
    # the discontinuity around the origin is it can never fling you away
    # hence you just get stuck at a distance of eps from the origin.
    
    # This probably means that knowldge of the Lipschitz constant is needed theoretically
    # but not in practice...

    # The case for discrete time systems is a little different since then you can get proper kicks

    return systems.uncontrolled_LTI_with_instability(A_mat, init_dist, condition)

def discrete_test():
    # DC_motor
    R= 2.0 # Ohms
    L= 0.5 # Henrys
    Km = .015 # torque constant
    Kb = .015 # emf constant
    Kf = 0.2 # Nms
    J= 0.02 # kg.m^2

    A_mat = 0.5*np.eye(2)

    init_mean = np.zeros(2)
    init_cov = np.eye(2)
    init_dist = scipy.stats.multivariate_normal(init_mean, init_cov).rvs

    return systems.uncontrolled_discrete_LTI(A_mat, init_dist, 0.1)

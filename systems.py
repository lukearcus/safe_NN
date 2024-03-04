import scipy
import numpy as np

class __base_system:

    def return_trajectory(self):
        pass

    def sample_init_state(self):
        pass

class __continuous_system(__base_system):
    
    def get_derivative(self, time, state):
        pass

    def return_trajectory(self, time_horizon):
        x_0 = self.sample_init_state()
        if x_0.ndim > 1:
            x_0 = x_0.flatten()
        traj = scipy.integrate.solve_ivp(self.get_derivative, (0, time_horizon), x_0, dense_output=True)
        state_traj = traj["y"]
        times = traj["t"]
        # Following will NOT return true derivatives for trajectory when stochasticity is included
        derivs = [self.get_derivative(time, state) for time, state in zip(times, state_traj.T)]

        return times, state_traj, derivs

class __discrete_system(__base_system):
    T = None

    def get_next_state(self, time, state):
        pass

    def return_trajectory(self, time_horizon):
        x_0 = self.sample_init_state()
        if x_0.ndim > 1:
            x_0 = x_0.flatten()
        traj = [x_0]
        for t in range(0, time_horizon, self.T):
            traj.append(self.get_next_state(t, traj[-1]))
        
        return traj

class uncontrolled_LTI(__continuous_system):
    A = None
    def __init__(self, _A, init_state_distribution):
        self.A = _A
        self.n_states = _A.shape[0]
        self.sample_init_state = init_state_distribution

    def get_derivative(self, time, state):
        if state.ndim == 1:
            state = np.expand_dims(state, axis=1)
        elif state.shape == (1,self.n_states):
            state = state.T
        return (self.A @ state).flatten()

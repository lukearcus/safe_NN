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

    def return_trajectory(self):
        x_0 = self.sample_init_state()
        if x_0.ndim > 1:
            x_0 = x_0.flatten()
        traj = scipy.integrate.solve_ivp(self.get_derivative, (0, self.time_horizon), x_0)
        state_traj = traj["y"]
        times = traj["t"]
        # Following will NOT return true derivatives for trajectory when stochasticity is included
        derivs = [self.get_derivative(time, state) for time, state in zip(times, state_traj.T)]
        return times, state_traj, derivs

class __discrete_system(__base_system):
    T = None

    def get_next_state(self, time, state):
        pass

    def return_trajectory(self):
        x_0 = self.sample_init_state()
        if x_0.ndim > 1:
            x_0 = x_0.flatten()
        traj = [x_0]
        times = []
        next_s = []
        for t in np.arange(0, self.time_horizon, self.T):
            next_state = self.get_next_state(t, traj[-1])
            traj.append(next_state)
            times.append(t)
            next_s.append(next_state)
        traj.pop(-1)
        return times, traj, next_s

class uncontrolled_discrete_LTI(__discrete_system):
    A = None
    def __init__(self, _A, init_state_distribution, _T, time_horizon):
        self.A = _A
        self.n_states = _A.shape[0]
        self.sample_init_state = init_state_distribution
        self.T = _T
        self.time_horizon = time_horizon
    
    def get_next_state(self, time, state):
        if state.ndim == 1:
            state = np.expand_dims(state, axis=1)
        elif state.shape == (1,self.n_states):
            state = state.T
        return (self.A @ state).flatten()

class uncontrolled_LTI(__continuous_system):
    A = None
    def __init__(self, _A, init_state_distribution, time_horizon):
        self.A = _A
        self.n_states = _A.shape[0]
        self.sample_init_state = init_state_distribution
        self.time_horizon = time_horizon

    def get_derivative(self, time, state):
        if state.ndim == 1:
            state = np.expand_dims(state, axis=1)
        elif state.shape == (1,self.n_states):
            state = state.T
        return (self.A @ state).flatten()

class uncontrolled_LTI_with_instability(uncontrolled_LTI):
    def __init__(self, _A, init_state_distribution, instability_test_func):
        super().__init__(_A, init_state_distribution)
        self.test = instability_test_func

    def get_derivative(self, time, state):
        if self.test(time, state):
            return 1e100*state/np.linalg.norm(state)
        else:
            return super().get_derivative(time, state)

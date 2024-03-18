import models
import pickle

num_traj = 100
model = models.get_simple_test()
trajectories = []
for i in range(num_traj):
    times, states, derivs = model.return_trajectory(10)
    trajectories.append((states, derivs))

with open("trajectory_data.pkl", 'wb') as f:
    pickle.dump(trajectories, f)

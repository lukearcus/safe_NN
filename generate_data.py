import models
import pickle

num_traj = 500
discrete = 0

if not discrete:
    model = models.get_simple_test()
    trajectories = []
    for i in range(num_traj):
        times, states, derivs = model.return_trajectory(10)
        trajectories.append((states, derivs))
    
    with open("trajectory_data.pkl", 'wb') as f:
        pickle.dump(trajectories, f)
else:
    model=models.discrete_test()
        
    trajectories = []
    for i in range(num_traj):
        times, states, nexts = model.return_trajectory(10)
        trajectories.append((states, nexts))
    
    with open("disc_trajectory_data.pkl", 'wb') as f:
        pickle.dump(trajectories, f)

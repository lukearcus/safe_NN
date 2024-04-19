import models
import networks
import matplotlib.pyplot as plt
import torch
import numpy as np
import trainer
import verifier
from tqdm import tqdm
import time
import pickle

torch.manual_seed(0)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

num_traj = 100
training_loops_per_run = 100
max_eps = 0.1
TimeOut = 120
empirical_samples = 1000
beta = 1e-5
model = models.discrete_test()

with open("disc_trajectory_data.pkl", 'rb') as f:
    data = pickle.load(f)

trajectories = data
#import pdb; pdb.set_trace() #test repeatability, works as intended!
#trajectories = trajectories[1:]
net = networks.structural_lyapunov().to(device)
for k in tqdm(range(training_loops_per_run)):
    trainer.train_disc_lyap(trajectories, net, device)           #start_time = time.perf_counter()
                                                            #trajectories = []
eps = verifier.verify_disc_lyap(trajectories, net, device, beta) #while time.perf_counter() - start_time < TimeOut:
#
#    for i in range(num_traj):
#        times, states, derivs = model.return_trajectory(10)
#        trajectories.append((states, derivs))
#    
#    net = networks.test_NN().to(device)
#    for k in tqdm(range(training_loops_per_run)):
#        trainer.train_lyap(trajectories, net, device)
#    
#    eps = verifier.verify_lyap(trajectories, net, device, beta)
#    if eps < max_eps:
#        break
#    num_traj *= 2
#if eps > max_eps:
#    print("Timed out")
empirical_eps, converge_eps = verifier.MC_test_disc_lyap(empirical_samples, net, device, model)
print(("Calculated upper bound on violation probability (with confidence {:.3f}: {:.3f}\n" + 
        "Empirical violation rate of lyapunov function: {:.3f}\n" +
        "Empirical rate of non-converged trajectories: {:.3f}")
        .format(beta, eps, empirical_eps, converge_eps))

states_leq_0 = 0
derivs_geq_0 = 0
x = np.arange(-2,2,0.01)
y = np.arange(-2,2,0.01)
X, Y = np.meshgrid(x, y)

lyap = []
for x in np.arange(-2,2,0.01):
    lyap_x = []
    for y in np.arange(-2,2,0.01):
        state = np.array([x,y]).T
        next_s = model.get_next_state(0,state)
        state, next_s = torch.from_numpy(state), torch.from_numpy(np.array(next_s))
        state, next_s = state.to(device, dtype=torch.float32), next_s.to(device, dtype=torch.float32)
        val = net(state)
        lyap_x.append(val.detach().item())
        lyap_deriv = net(next_s) - val
        if val < 0:
            states_leq_0 += 1
        if lyap_deriv > 0:
            derivs_geq_0 += 1
    lyap.append(lyap_x)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, np.array(lyap))
plt.savefig("fig_disc_test")
#print(traj)
#plt.plot(traj["t"], traj["y"][0])

import models
import networks
import matplotlib.pyplot as plt
import torch
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = models.get_simple_test()
for i in range(10):
    times, states, derivs = model.return_trajectory(10)
    
    net = networks.test_NN().to(device)
    networks.train_lyap(list(zip(states.T, derivs)), net, device)

states_leq_0 = 0
derivs_geq_0 = 0
x = np.arange(-2,2,0.1)
y = np.arange(-2,2,0.1)
X, Y = np.meshgrid(x, y)

lyap = []
for x in np.arange(-2,2,0.1):
    lyap_x = []
    for y in np.arange(-2,2,0.1):
        state = np.array([x,y])
        deriv = model.get_derivative(0,state)
        state, deriv = torch.from_numpy(state), torch.from_numpy(np.array(deriv))
        state, deriv = state.to(device, dtype=torch.float32), deriv.to(device, dtype=torch.float32)
        val = net(state)
        lyap_x.append(val.detach().item())
        lyap_deriv = net.get_deriv(state, deriv)
        if val < 0:
            states_leq_0 += 1
        if lyap_deriv > 0:
            derivs_geq_0 += 1
    lyap.append(lyap_x)
import pdb; pdb.set_trace()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, np.array(lyap))
plt.savefig("test")
import pdb; pdb.set_trace()
#print(traj)
#plt.plot(traj["t"], traj["y"][0])

import models
import matplotlib.pyplot as plt

model = models.get_simple_test()
times, states, derivs = model.return_trajectory(10)

import pdb; pdb.set_trace()
print(traj)
plt.plot(traj["t"], traj["y"][0])
plt.savefig("test")

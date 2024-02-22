import models
import matplotlib.pyplot as plt

model = models.get_simple_test()
traj = model.return_trajectory(10)

print(traj)
plt.plot(traj["t"], traj["y"][0])
plt.savefig("test")

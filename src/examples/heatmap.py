import matplotlib.pyplot as plt
from src import data
import numpy as np
import seaborn


trajs1 = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=300)

traj_list = [data.preprocess2(raw_traj) for raw_traj in trajs1]
subplot = plt.subplot()
'''
subplot.plot(traj_list[0][0:,0],traj_list[0][0:,1],marker='.')
subplot.plot(traj_list[1][64:124,0],traj_list[1][64:124,1],marker='.')
plt.show()
'''
trajs2 = data.load_files(r"assets/TNA-WUH", usecols=[7, 8, 5],num=300)
trajs3 = data.load_files(r"assets/SHA_CAN", usecols=[7, 8, 5],num=300)

traj_list = trajs1 + trajs2 + trajs3
rstacked_traj = np.row_stack(traj_list)

lons = rstacked_traj[:,0]
lats = rstacked_traj[:,1]
bin1 = int(np.around(max(lons)-min(lons),5)/0.025)
bin2 = int(np.around(max(lats)-min(lats),5)/0.025)

hist, lon, lat = np.histogram2d(lats, lons, [bin2,bin1])

ax = seaborn.heatmap(hist, robust=True, square=True)
plt.show()

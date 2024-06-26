import numpy as np
from dtw import *
import matplotlib.pyplot as plt
import data
raw_trajs = data.load_files(r"assets/PEK-SHA", usecols=[7, 8, 5],num=3)
traj_list = [data.preprocess2(raw_traj) for raw_traj in raw_trajs]
traj0 = traj_list[0][:,:2]
traj1 = traj_list[1][:,:2]
## A noisy sine wave as query
idx = np.linspace(0,6.28,num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula

alignment = dtw(traj1, traj0, step_pattern='asymmetric')

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
#dtw(query, template, keep_internals=True,
#    step_pattern=rabinerJuangStepPattern(6, "c"))\
#    .plot(type="twoway")

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
#rabinerJuangStepPattern(6,"c").plot()
plt.show()
## And much more!
import data
from classify import gridding
import numpy as np
from classify.dataset import TrajDataset
from classify.module import GridEncoder,SeqentialClassifier
from torch.utils.data.dataloader import DataLoader

raw_trajs1 = data.load_files("H:/TrajectoryPrediction/adsb/PEK-SHA/MU5122", usecols=[7, 8, 5], num=300)
raw_trajs2 = data.load_files("H:/TrajectoryPrediction/adsb/TNA-WUH", usecols=[7, 8, 5], num=300)
stacked_traj = np.row_stack(raw_trajs1 + raw_trajs2)[:, :2]
trajs1 = [data.preprocess(raw_traj)[:,:2] for raw_traj in raw_trajs1]
trajs2 = [data.preprocess(raw_traj)[:,:2] for raw_traj in raw_trajs2]
dataset = TrajDataset([trajs1,trajs2])
dataloader = DataLoader(dataset,shuffle=True)
grid_info = gridding.get_grid_info_by_len(stacked_traj,(0.02,0.02))
net = nn.Sequential(GridEncoder(grid_info, (8, 8)),
                    SeqentialClassifier(16, 16, 2))

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=0.001)
for epoch in range(1):
    for traj, label in dataloader:
        pred_label = net.forward(traj.squeeze())
        loss = loss_fn(pred_label, label.squeeze())
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(loss.item(),epoch)

test_traj1,_ = dataset.__getitem__(0)
test_traj2,_ = dataset.__getitem__(305)

pred_label1 = net(test_traj1).detach().numpy()
pred_label2 = net(test_traj2).detach().numpy()

print(0)
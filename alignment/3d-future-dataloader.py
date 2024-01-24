import numpy as np
import warnings
import os
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    pmax = np.max(pc, axis=0)
    pmin = np.min(pc, axis=0)
    centroid = (pmax + pmin) / 2.0
    return - centroid


class DataLoader(Dataset):
    def __init__(self, root, npoint=2048, split='train'):
        self.npoints = npoint
        self.split = split
        self.datalist = []
        with open(os.path.join(root, 'split', self.split + '_list.txt'), 'r') as f1:
            datafile = f1.readlines()
        for i in range(len(datafile)):
            if len(datafile[i]) > 1:
                # data_path = os.path.join(root, 'pc', datafile[i][:-1], 'pc', 'complete.pts')
                # data_path = os.path.join(root, 'pc', datafile[i][:-1], 'pc', 'complete.npy')
                data_path = os.path.join(root, 'pc', datafile[i][:-1], 'pc', 'part.npy')
                self.datalist.append(str(data_path))
        print('The size of %s data is %d' % (split, len(self.datalist)))

    def __getitem__(self, index):
        pc_file_path = self.datalist[index]
        pc_compelet = np.load(pc_file_path)
        # pc_compelet = np.load('./pre-process/a.npy')
        np.random.shuffle(pc_compelet)
        pc_sampled = pc_compelet[0:self.npoints, 0:3]
        color_sampled = pc_compelet[0:self.npoints, 3:6]
        normal_sampled = pc_compelet[0:self.npoints, 6:9]
        # normal to center at 0.
        # pc_sampled = pc_normalize(pc_sampled)
        return pc_sampled, color_sampled, normal_sampled

    def __len__(self):
        return len(self.datalist)

    def get_name(self, index):
        return self.ptss_name[index].decode('utf-8')

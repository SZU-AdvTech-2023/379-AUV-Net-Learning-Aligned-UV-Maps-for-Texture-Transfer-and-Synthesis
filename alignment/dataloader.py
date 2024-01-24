import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from pathlib import Path
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
		root_parent = str(Path(root).parent)
		with open(os.path.join(root_parent, 'split', self.split+'_list_auv.txt'), 'r') as f1:
			datafile = f1.readlines()
		for i in range(len(datafile)):
			if len(datafile[i]) > 1:
				# data_path = os.path.join(root, 'pc', datafile[i][:-1], 'pc', 'complete.pts')
				# data_path = os.path.join(root, 'pc', datafile[i][:-1], 'pc', 'complete.npy')
				data_path = os.path.join(root, datafile[i][:-1], 'pc', 'complete.npy')
				self.datalist.append(str(data_path))
		if split == 'train' and len(self.datalist) > 200:
			self.datalist = self.datalist[0:200]
			print('!!!!!!!!!!!!!!!')
		if split == 'val' and len(self.datalist) > 20:
			self.datalist = self.datalist[0:20]
			print('!!!!!!!!!!!!!!!')
		print('The size of %s data is %d'%(split, len(self.datalist)))

	def __getitem__(self, index):
		pc_file_path = self.datalist[index]
		pc_compelet = np.load(pc_file_path)
		np.random.shuffle(pc_compelet)
		pc_sampled = pc_compelet[0:self.npoints, 0:3]
		Ry = np.array([[np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
						[0, 1, 0],
						[-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]])
		color_sampled = pc_compelet[0:self.npoints, 3:6]
		
		normal_sampled = np.concatenate([pc_compelet[0:self.npoints,8:9], pc_compelet[0:self.npoints, 7:8], 1-pc_compelet[0:self.npoints,6:7]], axis=-1)
		pc_sampled = np.dot(pc_sampled, Ry).astype(np.float32)

		normal_sampled = (normal_sampled * 2) - 1
		return pc_sampled, color_sampled, normal_sampled, pc_file_path
		
	def __len__(self):
		return len(self.datalist)

	def get_name(self, index):
		return self.ptss_name[index].decode('utf-8')


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

def makePlyFile(xyzs, RGBS=None, normals=None, fileName='makeply.ply'):
    '''Make a ply file for open3d.visualization.draw_geometries
    :param xyzs:    numpy array of point clouds 3D coordinate, shape (numpoints, 3).
    :param labels:  numpy array of point label, shape (numpoints, ).
    '''
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if normals is not None:
            f.write('property float nx\n')
            f.write('property float ny\n')
            f.write('property float nz\n')
        if RGBS is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        if normals is not None and RGBS is not None:
            for i in range(len(xyzs)):
                r, g, b = RGBS[i]
                x, y, z = xyzs[i]
                nx, ny, nz = normals[i]
                f.write('{} {} {} {} {} {} {} {} {}\n'.format(x, y, z, nx, ny, nz, r, g, b))
        elif RGBS is not None:
            for i in range(len(xyzs)):
                r, g, b = RGBS[i]
                x, y, z = xyzs[i]
                f.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))
        elif normals is not None:
            for i in range(len(xyzs)):
                x, y, z = xyzs[i]
                nx, ny, nz = normals[i]
                f.write('{} {} {} {} {} {}\n'.format(x, y, z, nx, ny, nz))
        else:
            for i in range(len(xyzs)):
                x, y, z = xyzs[i]
                f.write('{} {} {}\n'.format(x, y, z))


class DataLoader(Dataset):
    def __init__(self, root, npoint=2048, split='train'):
        self.npoints = npoint
        self.split = split
        self.datalist = []
        with open(os.path.join(root, '../', 'chair.txt'), 'r') as f1:
            datafile = f1.readlines()
        # if split == 'train':
        #     datafile = datafile[:250]
        # elif split == 'val':
        #     datafile = datafile[250:280]
        # else:
        datafile = datafile[:330]
        for i in range(len(datafile)):
            if len(datafile[i]) > 1:
                # data_path = os.path.join(root, 'pc', datafile[i][:-1], 'pc', 'complete.pts')
                # data_path = os.path.join(root, 'pc', datafile[i][:-1], 'pc', 'complete.npy')
                data_path = os.path.join(root, datafile[i].replace('\n', '').replace('.png', '.npy'))
                self.datalist.append(str(data_path))
        print('The size of %s data is %d' % (split, len(self.datalist)))

    def __getitem__(self, index):
        pc_file_path = self.datalist[index]
        pc_compelet = np.load(pc_file_path)

        np.random.shuffle(pc_compelet)
        pc_sampled = pc_compelet[0:self.npoints, 0:3]
        color_sampled = pc_compelet[0:self.npoints, 3:6]
        normal_sampled = pc_compelet[0:self.npoints, 6:9]

        normal_sampled = normal_sampled * 2 - 1
        # input normal => [-1, 1]
        # shapes: facing x direction, top is y direction
        # normalized in a unit cube centered at origin

        return pc_sampled, color_sampled, normal_sampled, self.datalist[index]

    def __len__(self):
        return len(self.datalist)

    def get_name(self, index):
        return self.ptss_name[index].decode('utf-8')

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        N, C = xyz.shape
        centroids = np.zeros(npoint, dtype=np.longlong)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N, dtype=np.longlong)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :].reshape(1, 3)
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        return centroids
"""
auv-net to align textures.
"""
import sys

sys.path.append('../../code')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.nn import Sigmoid, Softmax
from pytorch3d.ops import add_points_features_to_volume_densities_features as voxelize

class shape_encoder(nn.Module):
    # checkd.
    def __init__(self, gen_num, basis_num):
        super(shape_encoder, self).__init__()
        self.gen_num = gen_num
        self.basis_num = basis_num
        self.conv3d1 = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm3d([32, 32, 32, 32])
        self.conv3d2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm3d([64, 16, 16, 16])
        self.conv3d3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm3d([128, 8, 8, 8])
        self.conv3d4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.in4 = nn.InstanceNorm3d([256, 4, 4, 4])
        self.conv3d5 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0)
        # instance norm can't be used to size 1 * 1 * 1.
        # self.in5 = nn.InstanceNorm3d([512,1,1,1])
        # self.conv3d6 = nn.Conv3d(in_channels=512, out_channels=512 + 1024 * 2, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(512, 256 + gen_num * basis_num * 9)

    def forward(self, pcs, colors):
        batch_size = pcs.size()[0]
        voxels = voxelize(points_3d=pcs, points_features=colors,
                          volume_densities=torch.zeros(batch_size, 1, 64, 64, 64).cuda(),
                          volume_features=torch.zeros(batch_size, 3, 64, 64, 64).cuda())
        voxels = torch.cat([voxels[1], voxels[0]], 1)
        fea1 = self.conv3d1(voxels)
        fea1 = F.leaky_relu(self.in1(fea1))
        fea2 = self.conv3d2(fea1)
        fea2 = F.leaky_relu(self.in2(fea2))
        fea3 = self.conv3d3(fea2)
        fea3 = F.leaky_relu(self.in3(fea3))
        fea4 = self.conv3d4(fea3)
        fea4 = F.leaky_relu(self.in4(fea4))
        fea5 = self.conv3d5(fea4)
        fea5 = fea5.squeeze()
        if batch_size == 1:
            fea5 = fea5.unsqueeze(dim=0)
        fea6 = F.leaky_relu(self.fc1(fea5))
        shape_code = fea6[:, 0:256]
        coeffs = fea6[:, 256:]
        coeffs = coeffs.chunk(self.gen_num, dim=1)
        return shape_code, coeffs


class shape_masker(nn.Module):
    # checked.
    def __init__(self):
        super(shape_masker, self).__init__()
        self.fc1 = nn.Linear(256 + 6, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, pcs, normals, shape_code):
        pcs_normals = torch.concat([pcs, normals], dim=2)
        shape_codes = shape_code.unsqueeze(dim=1).repeat(1, pcs_normals.size()[1], 1)
        pc_input = torch.concat([pcs_normals, shape_codes], dim=2)
        fea1 = F.leaky_relu(self.fc1(pc_input))
        fea2 = F.leaky_relu(self.fc2(fea1))
        fea3 = F.leaky_relu(self.fc3(fea2))
        mask = F.sigmoid(self.fc4(fea3))
        return mask

class chair_shape_masker(nn.Module):
    # checked.
    def __init__(self):
        super(chair_shape_masker, self).__init__()
        self.fc1 = nn.Linear(256 + 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_1 = nn.Linear(512, 1)
        self.fc4_2 = nn.Linear(512, 3)

    def forward(self, pcs, shape_code):
        pcs_ = pcs
        shape_codes = shape_code.unsqueeze(dim=1).repeat(1, pcs_.size()[1], 1)
        pc_input = torch.concat([pcs_, shape_codes], dim=2)
        fea1 = F.leaky_relu(self.fc1(pc_input))
        fea2 = F.leaky_relu(self.fc2(fea1))
        fea3 = F.leaky_relu(self.fc3(fea2))
        mask = F.sigmoid(self.fc4_1(fea3))
        normal = self.fc4_2(fea3)
        return normal, mask

class uv_mapper(nn.Module):
    # checked.
    def __init__(self):
        super(uv_mapper, self).__init__()
        self.fc1 = nn.Linear(256 + 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 2)

    def forward(self, pcs, shape_code):
        shape_codes = shape_code.unsqueeze(dim=1).repeat(1, pcs.size()[1], 1)
        pc_input = torch.concat([pcs, shape_codes], dim=2)
        fea1 = F.leaky_relu(self.fc1(pc_input))
        fea2 = F.leaky_relu(self.fc2(fea1))
        fea3 = F.leaky_relu(self.fc3(fea2))
        uv = self.fc4(fea3)
        return uv


class basis_generator(nn.Module):
    # checked.
    def __init__(self, G_dim, N):
        # G_dim is mid-feature dim, N is output channel.
        super(basis_generator, self).__init__()
        self.G_dim = G_dim
        self.N = N
        self.fc1 = nn.Linear(2, self.G_dim)
        self.fc2 = nn.Linear(self.G_dim + 2, self.G_dim)
        self.fc3 = nn.Linear(self.G_dim + 2, self.G_dim)
        self.fc4 = nn.Linear(self.G_dim + 2, self.G_dim)
        self.fc5 = nn.Linear(self.G_dim, self.G_dim)
        self.fc6 = nn.Linear(self.G_dim, self.G_dim)
        self.fc7 = nn.Linear(self.G_dim, self.N)

    def forward(self, uv):
        fea1 = F.leaky_relu(self.fc1(uv))
        fea1 = torch.concat([fea1, uv], dim=2)
        fea2 = F.leaky_relu(self.fc2(fea1))
        fea2 = torch.concat([fea2, uv], dim=2)
        fea3 = F.leaky_relu(self.fc3(fea2))
        fea3 = torch.concat([fea3, uv], dim=2)
        fea4 = F.leaky_relu(self.fc4(fea3))
        fea5 = F.leaky_relu(self.fc5(fea4))
        fea6 = F.leaky_relu(self.fc6(fea5))
        basis = self.fc7(fea6)
        return basis


class auv_net(nn.Module):
    def __init__(self, arg):
        super(auv_net, self).__init__()
        self.args = arg
        self.net = torch.nn.ModuleDict(self.initialize_networks())

    def initialize_networks(self):
        net = {}
        net['shape_encoder'] = shape_encoder(gen_num=self.args.num_generator, basis_num=self.args.num_N)
        net['shape_masker'] = chair_shape_masker()
        net['uv_mapper'] = uv_mapper()
        for i in range(self.args.num_generator):
            name = 'basis_generator_' + str(i)
            net[name] = basis_generator(G_dim=self.args.num_g_dim, N=self.args.num_N)
        if self.args.continue_train or self.args.align_texture:
            prefix = 'least' if self.args.align_texture else 'least'
            net['shape_encoder'] = utils.load_network(net['shape_encoder'], 'shape_encoder', self.args, prefix)
            net['shape_masker'] = utils.load_network(net['shape_masker'], 'shape_masker', self.args, prefix)
            net['uv_mapper'] = utils.load_network(net['uv_mapper'], 'uv_mapper', self.args, prefix)
            for i in range(self.args.num_generator):
                net['basis_generator_' + str(i)] = utils.load_network(net['basis_generator_' + str(i)],
                                                                      'basis_generator_' + str(i), self.args, prefix)
        return net

    def create_optimizer(self):
        optimizer_encoder = torch.optim.Adam(list(self.net['shape_encoder'].parameters()),
                                             lr=self.args.learning_rate,
                                             betas=(0.9, 0.999),
                                             eps=1e-05,
                                             weight_decay=self.args.decay_rate)
        optimizer_masker = torch.optim.Adam(list(self.net['shape_masker'].parameters()),
                                            lr=self.args.learning_rate,
                                            betas=(0.9, 0.999),
                                            eps=1e-05,
                                            weight_decay=self.args.decay_rate)
        optimizer_mapper = torch.optim.Adam(list(self.net['uv_mapper'].parameters()),
                                            lr=self.args.learning_rate,
                                            betas=(0.9, 0.999),
                                            eps=1e-05,
                                            weight_decay=self.args.decay_rate)
        for i in range(self.args.num_generator):
            basis_generator = self.net['basis_generator_' + str(i)]
            if i == 0:
                genertor_para = list(basis_generator.parameters())
            else:
                genertor_para += list(basis_generator.parameters())

        optimizer_generator = torch.optim.Adam(genertor_para,
                                               lr=self.args.learning_rate,
                                               betas=(0.9, 0.999),
                                               eps=1e-05,
                                               weight_decay=self.args.decay_rate)
        return optimizer_encoder, optimizer_masker, optimizer_mapper, optimizer_generator

    def forward(self, coors, colors, normals, mode=None, shapeCode=None):
        batch_size = coors.shape[0]
        if shapeCode == None:
            shape_code, coeffs = self.net['shape_encoder'](coors, colors)
        else:
            shape_code = shapeCode
        uv = self.net['uv_mapper'](coors, shape_code)
        # uv = torch.clamp(uv, -0.5, 0.5)

        '''
        """ original """
        mask = self.net['shape_masker'](coors, normals, shape_code)
        
        sep_pcs = []
        for i in range(self.args.num_generator):
            resized_coeff = coeffs[i].reshape(batch_size, self.args.num_N, 9)
            basis = self.net['basis_generator_' + str(i)](uv)
            sep_pcs.append(torch.bmm(basis, resized_coeff))
        recon_pcs = sep_pcs[0] * mask + sep_pcs[1] * (1 - mask)
        '''

        """ for chair """
        pred_normal, mask = self.net['shape_masker'](coors, shape_code)
        second_mask = Sigmoid()(torch.sum(pred_normal * normals, dim=-1, keepdim=True))

        mask = [second_mask * mask, second_mask * (1 - mask), (1 - second_mask) * mask, (1 - second_mask) * (1 - mask)]
        mask = torch.cat(mask, dim=-1)
        # mask = Softmax(dim=-1)(mask)

        if mode == 'align':
            return shape_code, uv, mask
        sep_pcs = []
        for i in range(self.args.num_generator):
            resized_coeff = coeffs[i].reshape(batch_size, self.args.num_N, 9)
            basis = self.net['basis_generator_' + str(i)](uv)
            sep_pcs.append(torch.bmm(basis, resized_coeff))
            if i == 0:
                recon_pcs = sep_pcs[i] * mask[:, :, i:i+1]
            else:
                recon_pcs += sep_pcs[i] * mask[:, :, i:i+1]
        recon_coors, recon_colors, recon_normals = recon_pcs.chunk(3, dim=2)
        losses = self.loss([coors, colors, normals], [uv, mask], [recon_coors, recon_colors, recon_normals])

        return losses

    def loss(self, original, mid, recons):
        # checked.
        coors, colors, normals = original
        uv, mask = mid
        recon_coors, recon_colors, recon_normals = recons
        loss_coor = F.mse_loss(recon_coors, coors)
        loss_color = F.mse_loss(recon_colors, colors)
        loss_normal = F.mse_loss(recon_normals, normals)
        # smooth loss.
        # 1. random sample points from input point cloud.
        idx = torch.randperm(coors.shape[1])
        coors = coors[:, idx, :].view(coors.size())
        uv = uv[:, idx, :].view(uv.size())
        sel_coors = coors[:, 0:self.args.num_loss_points, :]
        sel_uv = uv[:, 0:self.args.num_loss_points, :]
        # 2. compute distance between all and selected points.
        dis3d = torch.cdist(sel_coors, coors)
        dis2d = torch.cdist(sel_uv, uv)
        # 3. compute smooth loss.
        dis = torch.abs(dis3d - dis2d)
        weight = torch.where(dis3d < self.args.dist_thre, torch.ones_like(dis3d), torch.zeros_like(dis3d))
        loss_smooth = torch.sum(dis * weight) / (self.args.num_loss_points * self.args.num_train_points)
        loss_prior = self.chairs_prior_loss(original, recons, mid)
        return [loss_prior, loss_smooth, loss_coor, loss_color, loss_normal]

    def save(self, epoch):
        utils.save_network(self.net['shape_encoder'], 'shape_encoder', epoch, self.args)
        utils.save_network(self.net['shape_masker'], 'shape_masker', epoch, self.args)
        utils.save_network(self.net['uv_mapper'], 'uv_mapper', epoch, self.args)
        for i in range(self.args.num_generator):
            utils.save_network(self.net['basis_generator_' + str(i)], 'basis_generator_' + str(i), epoch, self.args)

    def cars_prior_loss(self, original, recons, mid):
        # prior loss (sum of masker and mapper prior).
        coors, colors, normals = original
        uv, mask = mid
        coors_xy = coors[:, :, 0:2]
        norms_z = normals[:, :, 2:3]
        # 1. mapper prior
        loss_prior_mapper = torch.sum(torch.pow((coors_xy - uv), 2))
        # 2. masker prior
        masker_prior = torch.where(normals > 0.5, torch.ones_like(norms_z), torch.zeros_like(norms_z))
        loss_prior_masker = torch.sum(torch.pow((mask - masker_prior), 2))
        loss_prior = (loss_prior_mapper + loss_prior_masker) / self.args.num_train_points
        return loss_prior

    def chairs_prior_loss(self, original, recons, mid):
        coors, colors, normals = original
        uv, mask = mid

        # get real mask split
        max_y_cord, _ = torch.max(coors[:, :, 1], dim=1)
        y_ = coors[:, :, 1] - max_y_cord.unsqueeze(1) - 0.05
        points_ = torch.cat([coors[:, :, 0:1], y_.unsqueeze(2), coors[:, :, 2:]], dim=2)
        # divide according to front and back (shown in normal direction)
        mask_1 = torch.where(torch.sum(points_ * normals, dim=-1, keepdim=True) < 0, 1, 0)
        point_seat_mask_1 = torch.where(coors[:, :, 0] > 0, 1, 0)
        point_seat_mask_2 = torch.where(coors[:, :, 0] < 0.1, 1, 0)
        point_seat_mask_3 = torch.where(coors[:, :, 2] > -0.05, 1, 0)
        point_seat_mask_4 = torch.where(coors[:, :, 2] < 0.05, 1, 0)
        point_seat_mask = point_seat_mask_1 * point_seat_mask_2 * point_seat_mask_3 * point_seat_mask_4
        batch_size = point_seat_mask.shape[0]
        y_seat = []
        for i in range(batch_size):
            idx = torch.nonzero(point_seat_mask[i])[:, 0]
            y_seat.append(max(coors[i][idx][:, 1]).unsqueeze(0))
        y_seat = torch.cat(y_seat, dim=0).unsqueeze(1).unsqueeze(1)
        # divide according to y-position (divide from seat position)
        mask_2 = torch.where(coors[:, :, 1:2] > y_seat - 0.2, 1, 0)
        real_mask_split = [mask_1 * mask_2, mask_1 * (1 - mask_2), (1 - mask_1) * mask_2, (1 - mask_1) * (1 - mask_2)]
        real_mask_split = torch.cat(real_mask_split, dim=-1)
        self.real_mask = real_mask_split
        dist = torch.sqrt(coors[:, :, 0:1] ** 2 + coors[:, :, 2:3] ** 2 + 4 * (coors[:, :, 1:2] - y_seat) ** 2) \
               / torch.sqrt(coors[:, :, 0:1] ** 2 + coors[:, :, 2:3] ** 2)
        t_x, t_y = coors[:, :, 0:1].mul(dist), coors[:, :, 2:3].mul(dist)

        # prior loss (sum of masker and mapper prior).
        loss_prior_masker = torch.sum(torch.pow((mask - real_mask_split), 2), dim=-1)
        loss_prior_mapper = torch.sum(torch.pow((torch.cat([t_x, t_y], dim=-1) - uv), 2), dim=-1)
        loss_prior = torch.sum(loss_prior_mapper + loss_prior_masker, dim=-1) / self.args.num_train_points
        return loss_prior




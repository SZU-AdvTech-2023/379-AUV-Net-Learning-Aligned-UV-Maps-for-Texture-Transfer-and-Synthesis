"""
transfer materials in 3D shapes.
"""
import sys

sys.path.append('..')

import argparse
import numpy as np
import os
import random
import trimesh
import torch
import torch.nn.functional as F

from model import auv_net
from PIL import Image
from pathlib import Path
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('AUV-Net')
    # environment settings.
    parser.add_argument('--gpus', type=int, default=[0], help='specify gpu device [default: 0]')
    # training settings.
    parser.add_argument('--model', default='auv-net-13', help='model name [default: auv-net]')
    parser.add_argument('--continue_train', type=bool, default=False, help='whether continue train.')
    parser.add_argument('--align_texture', type=bool, default=True, help='whether align textures.')
    # network archi. settings.
    parser.add_argument('--num_generator', type=int, default=4, help='number of basis generators')
    parser.add_argument('--num_g_dim', type=int, default=512, help='number of generator middle dimention')
    parser.add_argument('--num_N', type=int, default=64, help='number of generator output channel')
    # texture setting.
    parser.add_argument('--size', type=int, default=512, help='size of texture atlas')
    # model dir.
    parser.add_argument('--mesh_dir', type=str, default='../../data/normalized_obj/',
                        help='path to find simplifed mesh.')
    parser.add_argument('--texture_dir', type=str,
                        default='../../aligned_texture_2/',
                        help='path to find aligned texture.')
    parser.add_argument('--save_dir', type=str, default='../../aligned_texture_2/transfer_results/',
                        help='path to find aligned texture.')
    # save setting
    parser.add_argument('--save_mask', type=bool, default=False, help='whether save masked shape.')
    parser.add_argument('--save_reconed_pc', type=bool, default=False, help='whether save reconstructed pc.')
    # other comments.
    parser.add_argument('--comments', type=str, default='', help='Additional comments')
    return parser.parse_args()


def main(args):
    def texture_transfer_3d(args, source, shapes, model):
        with torch.no_grad():
            model = model.eval()
            for shape in tqdm(shapes):
                # set save dir.
                if not os.path.exists(os.path.join(args.save_dir, shape)):
                    os.makedirs(os.path.join(args.save_dir, shape))
                # step1 - get shape's code.
                # data_dir = Path('../../data/shapenet_pc_new2/02933112/pc/')
                data_dir = Path('../../data/')
                if os.path.exists(os.path.join(data_dir, 'replace_' + shape + '.npy')):
                    points = np.load(os.path.join(data_dir, 'replace_' + shape + '.npy'))
                else:
                    points = np.load(os.path.join(data_dir, 'pc', shape + '_complete.npy'))
                points[:, 6:9] = points[:, 6:9] * 2 - 1
                np.random.shuffle(points)
                coors_ori, colors_ori, normals_ori = points[0:16384, 0:3], points[0:16384, 3:6], points[0:16384, 6:9]
                coors_ori, colors_ori, normals_ori = torch.from_numpy(coors_ori), torch.from_numpy(
                    colors_ori), torch.from_numpy(normals_ori)
                coors_ori, colors_ori, normals_ori = coors_ori.float().cuda(), colors_ori.float().cuda(), normals_ori.float().cuda()
                coors_ori, colors_ori, normals_ori = coors_ori.unsqueeze(dim=0), colors_ori.unsqueeze(
                    dim=0), normals_ori.unsqueeze(dim=0)
                shape_code, _, _ = model(coors_ori, colors_ori, normals_ori, mode='align')
                # step2 - load mesh and get points' uvs.
                mesh = trimesh.load(os.path.join(args.mesh_dir, shape + '.obj'), force='mesh')
                face_center = np.array(mesh.triangles_center)
                face_normal = np.array(mesh.face_normals) # * np.array([0.5, 0.5, -0.5]) + np.array([0.5, 0.5, 0.5])
                face_center, face_normal = torch.from_numpy(face_center), torch.from_numpy(face_normal)
                face_center, face_normal = face_center.float().cuda(), face_normal.float().cuda()
                face_center, face_normal = face_center.unsqueeze(dim=0), face_normal.unsqueeze(dim=0)

                _, _, face_mask = model(face_center, face_center, face_normal, mode='align', shapeCode=shape_code)
                _, idx = torch.max(face_mask, dim=-1, keepdim=True)
                top_front_mask = torch.where(idx == 0, torch.ones_like(idx), torch.zeros_like(idx)).cpu().numpy()
                top_back_mask = torch.where(idx == 1, torch.ones_like(idx), torch.zeros_like(idx)).cpu().numpy()
                bottom_front_mask = torch.where(idx == 2, torch.ones_like(idx), torch.zeros_like(idx)).cpu().numpy()
                bottom_back_mask = torch.where(idx == 3, torch.ones_like(idx), torch.zeros_like(idx)).cpu().numpy()

                vertices = np.array(mesh.vertices)
                vertices = torch.from_numpy(vertices)
                vertices = vertices.float().cuda()
                vertices = vertices.unsqueeze(dim=0)
                _, vertex_uv, _ = model(vertices, vertices, vertices, mode='align', shapeCode=shape_code)
                vertex_uv = (torch.clamp(vertex_uv, -1.0, 1.0) + 1.0) / 2
                vertex_uv = vertex_uv.squeeze().detach().cpu().numpy()

                vertices_top_front = np.unique(mesh.faces[np.argwhere(top_front_mask == 0).squeeze()])
                top_front_new_uv = vertex_uv[vertices_top_front]
                vertex_uv[vertices_top_front] = top_front_new_uv

                vertices_top_back = np.unique(mesh.faces[np.argwhere(top_back_mask == 0).squeeze()])
                top_back_new_uv = vertex_uv[vertices_top_back]
                top_back_new_uv[:, 0] += 1
                vertex_uv[vertices_top_back] = top_back_new_uv

                vertices_bottom_front = np.unique(mesh.faces[np.argwhere(bottom_front_mask == 0).squeeze()])
                bottom_front_new_uv = vertex_uv[vertices_bottom_front]
                bottom_front_new_uv[:, 0] += 2
                vertex_uv[vertices_bottom_front] = bottom_front_new_uv

                vertices_bottom_back = np.unique(mesh.faces[np.argwhere(bottom_back_mask == 0).squeeze()])
                bottom_back_new_uv = vertex_uv[vertices_bottom_back]
                bottom_back_new_uv[:, 0] += 3
                vertex_uv[vertices_bottom_back] = bottom_back_new_uv

                vertex_uv[:, 0] /= 4.

                # add texture to mesh.
                texture_img = Image.open(os.path.join(args.texture_dir, str(source.replace('.npy', '')) + '_inpaint1.jpg'))
                # flip texture image.
                texture_img_np = np.array(texture_img)
                texture_img_top_front_np = texture_img_np[:, 1024:1536, :]
                texture_img_top_back_np = texture_img_np[:, 512:1024, :]
                texture_img_bottom_front_np = texture_img_np[:, 0:512, :]
                texture_img_bottom_back_np = texture_img_np[:, 1536:2048, :]

                texture_img_top_front_np = np.flip(texture_img_top_front_np, axis=1)
                texture_img_top_back_np = np.flip(texture_img_top_back_np, axis=1)
                texture_img_bottom_front_np = np.flip(texture_img_bottom_front_np, axis=1)
                texture_img_bottom_back_np = np.flip(texture_img_bottom_back_np, axis=1)

                new_texture_img = np.concatenate([texture_img_top_front_np, texture_img_top_back_np, texture_img_bottom_front_np, texture_img_bottom_back_np], axis=1)
                new_texture_img = Image.fromarray(np.uint8(new_texture_img))
                text_vis = trimesh.visual.texture.TextureVisuals(uv=vertex_uv, image=new_texture_img)
                mesh.visual = text_vis
                _ = mesh.export(os.path.join(args.save_dir, shape, 'transfered.obj'))
                print('shape')

    '''MODEL LOADING'''
    model = auv_net(args).cuda()

    '''GET ALIGNED TEXTURE'''
    shapes = []
    with open('../../data/chair.txt', 'r') as f1:
        shape_lines = f1.readlines()
    for line in shape_lines:
        if len(line) > 1:
            shapes.append(line[0:-1].replace('.png', '_complete.npy'))
    shapes = os.listdir(Path(args.mesh_dir))

    random.shuffle(shapes)
    # source_shape = '2f2e54607ea04be4c93bb8ae72b9da71'
    # source_shape = '4c7cc0a0e83f995ad40c07d3c15cc681'
    # source_shape = '39b51f3a6e8447c3c8a1a154de62786'
    # source_shape = '4d0ee9779956b74490a9ce3e4b15521e'
    # target_shapes = shapes[0:50]
    source_shape = '0000230'

    target_shapes = ['0000596']
    # target_shapes.append(source_shape)
    '''CREATE DIR'''
    args.save_dir = os.path.join(args.save_dir, source_shape)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    '''TEXTURE TRANSFER'''
    texture_transfer_3d(args, source_shape, target_shapes, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)

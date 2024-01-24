"""
generate aligned textured mesh.
"""
import sys

sys.path.append('..')

import argparse
import cv2
import numpy as np
import os
# import open3d as o3d
import trimesh
import torch
import torch.nn.functional as F

from model import auv_net
from PIL import Image
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('AUV-Net')
    # environment settings.
    parser.add_argument('--gpus', type=int, default=[0], help='specify gpu device [default: 0]')
    # training settings.
    parser.add_argument('--model', default='auv-net-1', help='model name [default: auv-net]')
    parser.add_argument('--continue_train', type=bool, default=False, help='whether continue train.')
    parser.add_argument('--align_texture', type=bool, default=True, help='whether align textures.')
    # data settings.
    parser.add_argument('--mesh_path', type=str, default='../../data/shapenet/02933112/', help='path to find mesh')
    parser.add_argument('--texture_path', type=str, default='../../data/shapenet_pc_new2/02933112/',
                        help='path to find texture')
    # network archi. settings.
    parser.add_argument('--num_generator', type=int, default=2, help='number of basis generators')
    parser.add_argument('--num_g_dim', type=int, default=1024, help='number of generator middle dimention')
    parser.add_argument('--num_N', type=int, default=64, help='number of generator output channel')
    # texture setting.
    parser.add_argument('--size', type=int, default=256, help='size of texture atlas')
    # model dir.
    parser.add_argument('--mesh_dir', type=str, default='./textureMesh/shape/', help='path to find simplifed mesh.')
    parser.add_argument('--texture_dir', type=str, default='./textureMesh/texture/',
                        help='path to find aligned texture.')
    parser.add_argument('--save_dir', type=str, default='./textureMesh/newShape/', help='path to find aligned texture.')
    # save setting
    parser.add_argument('--save_mask', type=bool, default=False, help='whether save masked shape.')
    parser.add_argument('--save_reconed_pc', type=bool, default=False, help='whether save reconstructed pc.')
    # other comments.
    parser.add_argument('--comments', type=str, default='', help='Additional comments')
    return parser.parse_args()


def main(args):
    def texture_mesh(args, shapes, model):
        with torch.no_grad():
            model = model.eval()
            for shape in tqdm(shapes):
                # set save dir.
                if not os.path.exists(os.path.join(args.save_dir, shape)):
                    os.makedirs(os.path.join(args.save_dir, shape))
                # step1 - get shape's code.
                data_dir = Path('../../data/pc/')
                points = np.load(os.path.join(data_dir, shape.replace('.obj', '_complete.npy')))
                np.random.shuffle(points)
                coors_ori, colors_ori, normals_ori = points[0:16384, 0:3], points[0:16384, 3:6], points[0:16384, 6:9]
                Ry = np.array([[np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)],
                               [0, 1, 0],
                               [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])

                coors_ori = np.dot(coors_ori, Ry).astype(np.float32)
                coors_ori, colors_ori, normals_ori = torch.from_numpy(coors_ori), torch.from_numpy(
                    colors_ori), torch.from_numpy(normals_ori)
                coors_ori, colors_ori, normals_ori = coors_ori.float().cuda(), colors_ori.float().cuda(), normals_ori.float().cuda()
                coors_ori, colors_ori, normals_ori = coors_ori.unsqueeze(dim=0), colors_ori.unsqueeze(
                    dim=0), normals_ori.unsqueeze(dim=0)
                shape_code, _, _ = model(coors_ori, colors_ori, normals_ori, mode='align')
                # step2 - load mesh and get points' uvs.
                mesh = trimesh.load(os.path.join(args.mesh_dir, shape), force='mesh')

                mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
                face_center = np.array(mesh.triangles_center)
                face_normal = np.array(mesh.face_normals) * np.array([0.5, 0.5, -0.5]) + np.array([0.5, 0.5, 0.5])
                face_center, face_normal = torch.from_numpy(face_center), torch.from_numpy(face_normal)
                face_center, face_normal = face_center.float().cuda(), face_normal.float().cuda()
                face_center, face_normal = face_center.unsqueeze(dim=0), face_normal.unsqueeze(dim=0)
                _, _, face_mask = model(face_center, face_center, face_normal, mode='align', shapeCode=shape_code)

                """
                
                face_mask = torch.where(face_mask > 0.5, torch.ones_like(face_mask), torch.zeros_like(face_mask))

                vertices = np.array(mesh.vertices)
                vertices = torch.from_numpy(vertices)
                vertices = vertices.float().cuda()
                vertices = vertices.unsqueeze(dim=0)
                _, vertex_uv, _ = model(vertices, vertices, vertices, mode='align', shapeCode=shape_code)
                vertex_uv = torch.clamp(vertex_uv, -0.5, 0.5) + 0.5
                vertex_uv = vertex_uv.squeeze().detach().cpu().numpy()
                vertices_back = np.unique(mesh.faces[np.argwhere(face_mask == 0).squeeze()])
                vertices_back = np.unique(mesh.faces[top_front_mask])
                back_new_uv = vertex_uv[vertices_back]
                back_new_uv[:, 0] += 1
                vertex_uv[vertices_back] = back_new_uv
                vertex_uv[:, 0] /= 2.

                # add texture to mesh.
                texture_img = Image.open(os.path.join(args.texture_dir, str(shape) + '_inpaint.jpg'))
                # flip texture image.
                texture_img_np = np.array(texture_img)
                texture_img_front_np = texture_img_np[:, 0:512, :]
                texture_img_back_np = texture_img_np[:, 512:, :]
                texture_img_front_np = np.flip(texture_img_front_np, axis=1)
                texture_img_back_np = np.flip(texture_img_back_np, axis=1)
                new_texture_img = np.append(texture_img_front_np, texture_img_back_np, axis=1)
                """
                mask = torch.cat(face_mask, dim=-1)
                _, idx = torch.max(mask, dim=-1)
                top_front_mask = torch.where(idx == 0, torch.ones_like(idx), torch.zeros_like(idx))
                top_back_mask = torch.where(idx == 1, torch.ones_like(idx), torch.zeros_like(idx))
                bottom_front_mask = torch.where(idx == 2, torch.ones_like(idx), torch.zeros_like(idx))
                bottom_back_mask = torch.where(idx == 3, torch.ones_like(idx), torch.zeros_like(idx))

                top_front_mask = top_front_mask.squeeze().detach().cpu().numpy().astype(np.uint8)
                top_back_mask = top_back_mask.squeeze().detach().cpu().numpy().astype(np.uint8)
                bottom_front_mask = bottom_front_mask.squeeze().detach().cpu().numpy().astype(np.uint8)
                bottom_back_mask = bottom_back_mask.squeeze().detach().cpu().numpy().astype(np.uint8)

                vertices = np.array(mesh.vertices)
                vertices = torch.from_numpy(vertices)
                vertices = vertices.float().cuda()
                vertices = vertices.unsqueeze(dim=0)
                _, vertex_uv, _ = model(vertices, vertices, vertices, mode='align', shapeCode=shape_code)
                vertex_uv = torch.clamp(vertex_uv, -0.5, 0.5) + 0.5
                vertex_uv = vertex_uv.squeeze().detach().cpu().numpy()

                vertices_top_front = np.unique(mesh.faces[np.argwhere(top_front_mask == 1).squeeze()])
                top_front_new_uv = vertex_uv[vertices_top_front]
                top_front_new_uv[:, 0] += 1
                vertex_uv[vertices_top_front] = top_front_new_uv
                vertex_uv[:, 0] /= 2.

                # add texture to mesh.
                texture_img = Image.open(os.path.join(args.texture_dir, str(shape) + '_inpaint.jpg'))
                # flip texture image.
                texture_img_np = np.array(texture_img)
                texture_img_front_np = texture_img_np[:, 0:512, :]
                texture_img_back_np = texture_img_np[:, 512:, :]
                texture_img_front_np = np.flip(texture_img_front_np, axis=1)
                texture_img_back_np = np.flip(texture_img_back_np, axis=1)
                new_texture_img = np.append(texture_img_front_np, texture_img_back_np, axis=1)

                new_texture_img = Image.fromarray(np.uint8(new_texture_img))
                text_vis = trimesh.visual.texture.TextureVisuals(uv=vertex_uv, image=new_texture_img)
                mesh.visual = text_vis
                _ = mesh.export(os.path.join(args.save_dir, shape, 'test.obj'))
                print('shape')

    '''CREATE DIR'''
    # save_dir = Path(args.data_path).parent.joinpath('aligned_texture_biggerer_distortion')
    # save_dir.mkdir(exist_ok=True)

    '''MODEL LOADING'''
    model = auv_net(args).cuda()

    '''GET ALIGNED TEXTURE'''
    shapes = os.listdir(Path(args.mesh_dir))
    shapes.sort()
    shapes = shapes[0:2]
    # align_texture(args, shapes, model)
    texture_mesh(args, shapes, model)
    print('shape')


if __name__ == '__main__':
    args = parse_args()
    main(args)

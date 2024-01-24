import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import argparse
import cv2
import math
import numpy as np
import os
import torch
import torch.nn.functional as F

from pathlib import Path
from model import auv_net
from tqdm import tqdm

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
    # data settings.
    parser.add_argument('--data_path', type=str, default='/workspace/2023/AUV_Net/data/',
                        help='path to find dataset')
    # network archi. settings.
    parser.add_argument('--num_generator', type=int, default=4, help='number of basis generators')
    parser.add_argument('--num_g_dim', type=int, default=512, help='number of generator middle dimention')
    parser.add_argument('--num_N', type=int, default=64, help='number of generator output channel')
    # texture setting.
    parser.add_argument('--size', type=int, default=512, help='size of texture atlas')
    # save setting
    parser.add_argument('--save_mask', type=bool, default=False, help='whether save masked shape.')
    parser.add_argument('--save_reconed_pc', type=bool, default=False, help='whether save reconstructed pc.')
    # other comments.
    parser.add_argument('--comments', type=str, default='', help='Additional comments')
    return parser.parse_args()


def mutil_put(hw, pixel_coor, color):
    ori_image = torch.zeros((3, hw, hw)).cuda()
    try:
        index = [pixel_coor[:, 0].squeeze().long(), pixel_coor[:, 1].squeeze().long()]
        for i in range(3):
            temp_ori = ori_image[i, :, :]
            ori_image[i, :, :] = temp_ori.index_put(index, color[:, i])
    except:
        print('mistake')
    return ori_image


def multi_select(coor, mask):
    for i in range(coor.shape[2]):
        single_coor = coor[:, :, i].unsqueeze(dim=2)
        sel_coor = torch.masked_select(single_coor, mask)
        if i == 0:
            sel_coors = sel_coor.unsqueeze(dim=1)
        else:
            sel_coors = torch.cat((sel_coors, sel_coor.unsqueeze(dim=1)), dim=1)
    return sel_coors


def flood_inpaint(size, pixel_coor, color):
    imgs = {}
    masks = {}
    x = pixel_coor.clone().cpu().numpy()
    un, idx = np.unique(x[:, 0] * 512 + x[:, 1], return_index=True)
    pixel_coor = pixel_coor[idx]
    color = color[idx]
    img = mutil_put(size, pixel_coor, color)
    img = (img * 255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    imgs['ori'] = img
    mask = mutil_put(size, pixel_coor, torch.ones_like(color).float())
    mask = (mask * 255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)[:, :, 0]
    masks['ori'] = mask
    mask_flood = flood_mask(mask)
    add_mask = ((mask != mask_flood) * 255.).astype(np.uint8)
    img_flood = cv2.inpaint(img, add_mask, 3, cv2.INPAINT_TELEA)
    imgs['flood'] = img_flood
    masks['flood'] = mask_flood
    img_inpaint = cv2.inpaint(img, ((mask != 255) * 255.).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    imgs['inpaint'] = img_inpaint
    return imgs, masks


def save_reconed_pc(save_dir, shape, output):
    ori_pc = np.load(os.path.join(save_dir.parent, 'pc', shape, 'pc', 'complete.npy'))
    recon_coor, recon_color, recon_normal = output
    recon_coor = recon_coor.squeeze().cpu().numpy()
    recon_color = recon_color.squeeze().cpu().numpy()
    recon_normal = recon_normal.squeeze().cpu().numpy()
    recon_pc = np.concatenate([recon_coor, recon_color, recon_normal], axis=1)
    np.savez(os.path.join(save_dir, shape, 'pcs.npz'),
             ori=ori_pc, recon=recon_pc)


def flood_mask(ori_mask):
    flood_mask = ori_mask.copy()
    h = ori_mask.shape[0]
    mask = np.zeros((h + 2, h + 2), np.uint8)
    cv2.floodFill(flood_mask, mask, (0, 0), 255)
    flood_mask_inv = cv2.bitwise_not(flood_mask)
    mask_out = ori_mask | flood_mask_inv
    return mask_out


def transform_result(uv, mask, colors):
    size = 512
    # pre-process.
    uv = torch.clamp(uv, -1.0, 1.0)
    uv = (uv + 1.0) / 2
    pixel_coor = uv * (size - 1)

    _, idx = torch.max(mask, dim=-1, keepdim=True)
    top_front_mask = torch.where(idx == 0, torch.ones_like(idx), torch.zeros_like(idx))
    top_back_mask = torch.where(idx == 1, torch.ones_like(idx), torch.zeros_like(idx))
    bottom_front_mask = torch.where(idx == 2, torch.ones_like(idx), torch.zeros_like(idx))
    bottom_back_mask = torch.where(idx == 3, torch.ones_like(idx), torch.zeros_like(idx))

    # if args.save_reconed_pc:
    # 	save_reconed_pc(save_dir, shape, [recon_coors, recon_colors, recon_normals])
    pixel_coor_top_front = torch.round(multi_select(pixel_coor, (top_front_mask == 1))).long()
    pixel_coor_top_back = torch.round(multi_select(pixel_coor, (top_back_mask == 1))).long()
    pixel_coor_bottom_front = torch.round(multi_select(pixel_coor, (bottom_front_mask == 1))).long()
    pixel_coor_bottom_back = torch.round(multi_select(pixel_coor, (bottom_back_mask == 1))).long()

    color_top_front = multi_select(colors, (top_front_mask == 1))
    color_top_back = multi_select(colors, (top_back_mask == 1))
    color_bottom_front = multi_select(colors, (bottom_front_mask == 1))
    color_bottom_back = multi_select(colors, (bottom_back_mask == 1))

    # map to img space and get inpating masks.
    imgs_top_front, img_masks_top_front = flood_inpaint(size, pixel_coor_top_front, color_top_front)
    imgs_top_back, img_masks_top_back = flood_inpaint(size, pixel_coor_top_back, color_top_back)
    imgs_bottom_front, img_masks_bottom_front = flood_inpaint(size, pixel_coor_bottom_front, color_bottom_front)
    imgs_bottom_back, img_masks_bottom_back = flood_inpaint(size, pixel_coor_bottom_back, color_bottom_back)

    # save image.
    img_top_front_wo = np.flip(np.rot90(imgs_top_front['ori']), axis=1)
    img_top_front_flood = np.flip(np.rot90(imgs_top_front['flood']), axis=1)
    img_top_front_inpaint = np.flip(np.rot90(imgs_top_front['inpaint']), axis=1)

    img_top_back_wo = np.flip(np.rot90(imgs_top_back['ori']), axis=1)
    img_top_back_flood = np.flip(np.rot90(imgs_top_back['flood']), axis=1)
    img_top_back_inpaint = np.flip(np.rot90(imgs_top_back['inpaint']), axis=1)

    img_bottom_front_wo = np.flip(np.rot90(imgs_bottom_front['ori']), axis=1)
    img_bottom_front_flood = np.flip(np.rot90(imgs_bottom_front['flood']), axis=1)
    img_bottom_front_inpaint = np.flip(np.rot90(imgs_bottom_front['inpaint']), axis=1)

    img_bottom_back_wo = np.flip(np.rot90(imgs_bottom_back['ori']), axis=1)
    img_bottom_back_flood = np.flip(np.rot90(imgs_bottom_back['flood']), axis=1)
    img_bottom_back_inpaint = np.flip(np.rot90(imgs_bottom_back['inpaint']), axis=1)

    img_wo = np.concatenate([img_top_front_wo, img_top_back_wo, img_bottom_front_wo, img_bottom_back_wo], axis=1)
    img_flood = np.concatenate([img_top_front_flood, img_top_back_flood, img_bottom_front_flood, img_bottom_back_flood],
                               axis=1)
    img_inpaint = np.concatenate(
        [img_top_front_inpaint, img_top_back_inpaint, img_bottom_front_inpaint, img_bottom_back_inpaint], axis=1)

    mask_top_front_wo = np.flip(np.rot90(img_masks_top_front['ori']), axis=1)
    mask_top_front_flood = np.flip(np.rot90(img_masks_top_front['flood']), axis=1)
    mask_top_back_wo = np.flip(np.rot90(img_masks_top_back['ori']), axis=1)
    mask_top_back_flood = np.flip(np.rot90(img_masks_top_back['flood']), axis=1)
    mask_bottom_front_wo = np.flip(np.rot90(img_masks_bottom_front['ori']), axis=1)
    mask_bottom_front_flood = np.flip(np.rot90(img_masks_bottom_front['flood']), axis=1)
    mask_bottom_back_wo = np.flip(np.rot90(img_masks_bottom_back['ori']), axis=1)
    mask_bottom_back_flood = np.flip(np.rot90(img_masks_bottom_back['flood']), axis=1)

    mask_wo = np.concatenate([mask_top_front_wo, mask_top_back_wo, mask_bottom_front_wo, mask_bottom_back_wo], axis=1)
    mask_flood = np.concatenate(
        [mask_top_front_flood, mask_top_back_flood, mask_bottom_front_flood, mask_bottom_back_flood], axis=1)

    return [img_wo, img_flood, img_inpaint], [mask_wo, mask_flood]


def farthest_point_sample(xyz, npoint):
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


def main(args):
    def align_texture_shapeCode(args, shapes, model):
        with torch.no_grad():
            model = model.eval()
            for shape in tqdm(shapes):

                # get shape code first.
                data_dir = Path("/workspace/2023/AUV_Net/data/pc/")
                points = np.load(os.path.join(data_dir, shape + '_part.npy'))
                np.random.shuffle(points)
                coors_ori, colors_ori, normals_ori = points[:16384, 0:3], points[:16384, 3:6], points[:16384, 6:9]
                normals_ori = normals_ori * 2 - 1
                coors_ori, colors_ori, normals_ori = torch.from_numpy(coors_ori), torch.from_numpy(
                    colors_ori), torch.from_numpy(normals_ori)
                coors_ori, colors_ori, normals_ori = coors_ori.float().cuda(), colors_ori.float().cuda(), normals_ori.float().cuda()
                coors_ori, colors_ori, normals_ori = coors_ori.unsqueeze(dim=0), colors_ori.unsqueeze(
                    dim=0), normals_ori.unsqueeze(dim=0)
                shape_code, _, _ = model(coors_ori, colors_ori, normals_ori, mode='align')
                # np.save(os.path.join(save_dir, str(shape) + '_shapeCode.npy'), shape_code.squeeze().detach().cpu().numpy())

                # shape_code = shape_code.unsqueeze(dim=0).cuda()

                # CASE1:load complete point clouds.
                # # sample 40w points.
                data_dir = Path(args.data_path)
                points = np.load(os.path.join(data_dir, shape + '_complete.npy'))
                np.random.shuffle(points)
                points[:, 6:9] = points[:, 6:9] * 2 - 1
                all_coors, all_colors, all_normals = points[:, 0:3], points[:, 3:6], points[:, 6:9]

                all_colors = torch.from_numpy(all_colors).float().cuda().unsqueeze(dim=0)
                point_batch = 100000
                start = 0
                end = point_batch
                for batch in range(math.ceil(points.shape[0] / point_batch)):
                    coors, colors, normals = points[start:end, 0:3], points[start:end, 3:6], points[start:end, 6:9]
                    coors, colors, normals = torch.from_numpy(coors), torch.from_numpy(colors), torch.from_numpy(normals)
                    coors, colors, normals = coors.float().cuda(), colors.float().cuda(), normals.float().cuda()
                    coors, colors, normals = coors.unsqueeze(dim=0), colors.unsqueeze(dim=0), normals.unsqueeze(dim=0)
                    _, uv, mask = model(coors, colors, normals, mode='align', shapeCode=shape_code)
                    start += point_batch
                    end = end + point_batch if (end + point_batch) < points.shape[0] else points.shape[0]
                    if batch == 0:
                        uvs, masks = uv, mask
                    else:
                        uvs, masks = torch.cat((uvs, uv), dim=1), torch.cat((masks, mask), dim=1)
                uv, mask = uvs, masks
                colors = all_colors
                [img_wo, img_flood, img_inpaint], [mask_wo, mask_flood] = transform_result(uv, mask, colors)
                cv2.imwrite(os.path.join(save_dir, str(shape.replace('.npy', '')) + '_ori.jpg'), img_wo[..., ::-1])
                cv2.imwrite(os.path.join(save_dir, str(shape.replace('.npy', '')) + '_flood.jpg'), img_flood[..., ::-1])
                cv2.imwrite(os.path.join(save_dir, str(shape.replace('.npy', '')) + '_inpaint.jpg'), img_inpaint[..., ::-1])
                cv2.imwrite(os.path.join(save_dir, str(shape.replace('.npy', '')) + '_mask.png'), mask_wo)
                cv2.imwrite(os.path.join(save_dir, str(shape.replace('.npy', '')) + '_flood.png'), mask_flood)

    '''CREATE DIR'''
    save_dir = Path(args.data_path).parent.joinpath('aligned_texture_2')
    save_dir.mkdir(exist_ok=True)

    '''MODEL LOADING'''
    model = auv_net(args).cuda()

    '''GET ALIGNED TEXTURE'''
    shapes = os.listdir(Path(args.data_path))
    shapes.sort()
    shapes = ['0002359']
    # shapes = ['2f2e54607ea04be4c93bb8ae72b9da71']
    align_texture_shapeCode(args, shapes, model)
    print('shape')


if __name__ == '__main__':
    args = parse_args()
    main(args)

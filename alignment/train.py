import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import argparse
import logging
from statistics import mode
import numpy as np
import random
import shutil
import torch
import torch.nn.functional as F
import zipfile

# from my_dataloader import DataLoader
from dataloader import DataLoader
from pathlib import Path
from model import auv_net
from tqdm import tqdm
from utils import visualizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
batch_idx = 0


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('AUV-Net')
    # environment settings.
    parser.add_argument('--gpus', type=int, default=[0, 1], help='specify gpu device [default: 0]')
    # training settings.
    parser.add_argument('--batch_size', type=int, default=11, help='batch size in training [default: 2]')
    parser.add_argument('--model', default='debug', help='model name [default: auv-net]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=0.001, help='decay rate [default: 1e-4]')
    parser.add_argument('--adjust_weights', type=int, default=[0, 50, 100, 150], help='epochs to adjust loss weights.')
    parser.add_argument('--weights', type=float,
                        default=[[100, 100, 10, 1, 1], [0, 100, 10, 1, 1], [0, 10, 10, 1, 1], [0, 100, 100, 1, 1]],
                        help='the loss weights of prior, smooth, coor, color, normal')
    parser.add_argument('--num_save_cks', type=int, default=10, help='number of epochs to save checkpoints')
    parser.add_argument('--continue_train', type=bool, default=False, help='whether continue train.')
    parser.add_argument('--is_optimization', type=bool, default=False, help='whether it is on optimization stages.')
    parser.add_argument('--align_texture', type=bool, default=False, help='whether align textures.')
    # data settings.
    parser.add_argument('--data_path', type=str, default='../../data/xyp2/chair/matPC/',
                        help='path to find dataset')
    parser.add_argument('--out_path', type=str, default='../../data/xyp2/chair/alignment-net/', help='path to save cks')
    parser.add_argument('--num_train_points', type=int, default=16384, help='number of sampled points for training')
    # network archi. settings.
    parser.add_argument('--num_generator', type=int, default=4,
                        help='number of basis generators')  # chair 4 -- origin 2
    parser.add_argument('--num_g_dim', type=int, default=512,
                        help='number of generator middle dimention')  # chair 512 -- origin 1024
    parser.add_argument('--num_N', type=int, default=64, help='number of generator output channel')
    # loss settings.
    parser.add_argument('--num_loss_points', type=int, default=2048,
                        help='number of sampled points for compute SMOOTH loss')
    parser.add_argument('--dist_thre', type=float, default=0.02,
                        help='distance to sample a local region to compute SMOOTH loss')
    parser.add_argument('--num_vis', type=int, default=10, help='number of batches to vis losses during training')
    # other comments.
    parser.add_argument('--comments', type=str, default='', help='Additional comments')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    def one_epoch(mode, epoch, dataloader, model, optimizers, viser):
        loss_priors, loss_smooths, loss_colors, loss_normals, loss_coors = [], [], [], [], []
        optimizer_encoder, optimizer_masker, optimizer_mapper, optimizer_generator = optimizers
        if mode == 'train':
            model = model.train()
        else:
            model = model.eval()

        for batch_id, data in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
            global batch_idx
            # print(batch_idx)
            if mode == 'train':
                optimizer_encoder.zero_grad()
                optimizer_masker.zero_grad()
                optimizer_mapper.zero_grad()
                optimizer_generator.zero_grad()
            # prapare data.
            coors, colors, normals, path = data
            coors, colors, normals = coors.float().cuda(), colors.float().cuda(), normals.float().cuda()
            # forward.
            losses = model(coors, colors, normals)
            if len(args.gpus) > 1:
                for idx, loss in enumerate(losses):
                    losses[idx] = losses[idx].sum() / len(args.gpus)
            # compute loss.
            i = 0
            for idx in range(len(args.adjust_weights)):
                if (epoch >= args.adjust_weights[idx]):
                    i = idx
            loss_total = losses[0] * args.weights[i][0] + losses[1] * args.weights[i][1] + losses[2] * args.weights[i][
                2] + losses[3] * args.weights[i][3] + losses[4] * args.weights[i][4]
            # losses = [losses[0] * args.weights[i][0], losses[1] * args.weights[i][1], losses[2] * args.weights[i][2], losses[3] * args.weights[i][3], losses[4] * args.weights[i][4]]
            # backward.
            if mode == 'train':
                loss_total.backward()
                optimizer_encoder.step()
                optimizer_masker.step()
                optimizer_mapper.step()
                optimizer_generator.step()
                batch_idx += 1
            # save and vis [weighted] loss.
            for idx, loss in enumerate(losses):
                losses[idx] = losses[idx] * args.weights[i][idx]

            loss_priors.append(losses[0].item())
            loss_smooths.append(losses[1].item())
            loss_coors.append(losses[2].item())
            loss_colors.append(losses[3].item())
            loss_normals.append(losses[4].item())
        # if(batch_idx % args.num_vis == 0):
        # 	viser.vis_loss_train(batch_index=batch_idx, losses=losses)
        losses_epoch = [np.mean(loss_priors), np.mean(loss_smooths), np.mean(loss_coors), np.mean(loss_colors),
                        np.mean(loss_normals)]

        # visualize
        with torch.no_grad():
            from alignChairTexture import transform_result
            from torch.nn import Sigmoid
            shape_code, coeffs = model.module.net['shape_encoder'](coors[0:1], colors[0:1])
            uv = model.module.net['uv_mapper'](coors[0:1], shape_code)

            ### debug -- 训崩uv会变成nan
            print('uv: ', uv.min(), uv.max())
            if torch.isnan(uv).any():
                print(path)

            pred_normal, mask = model.module.net['shape_masker'](coors[0:1], shape_code)

            second_mask = Sigmoid()(torch.sum(pred_normal * normals[0:1], dim=-1, keepdim=True))
            mask = [second_mask * mask, second_mask * (1 - mask), (1 - second_mask) * mask,
                    (1 - second_mask) * (1 - mask)]
            mask = torch.cat(mask, dim=-1)

            sep_pcs = []
            for i in range(args.num_generator):
                resized_coeff = coeffs[i].reshape(1, args.num_N, 9)
                basis = model.module.net['basis_generator_' + str(i)](uv)
                sep_pcs.append(torch.bmm(basis, resized_coeff))
                if i == 0:
                    recon_pcs = sep_pcs[i] * mask[:, :, i:i + 1]
                else:
                    recon_pcs += sep_pcs[i] * mask[:, :, i:i + 1]
            recon_coors, recon_colors, recon_normals = recon_pcs.chunk(3, dim=2)

            max_y_cord, _ = torch.max(coors[0:1, :, 1], dim=1)
            y_ = coors[0:1, :, 1] - max_y_cord.unsqueeze(1) - 0.05
            points_ = torch.cat([coors[0:1, :, 0:1], y_.unsqueeze(2), coors[0:1, :, 2:]], dim=2)
            # divide according to front and back (shown in normal direction)
            mask_1 = torch.where(torch.sum(points_ * normals[0:1], dim=-1, keepdim=True) < 0, 1, 0)
            point_seat_mask_1 = torch.where(coors[0:1, :, 0] > 0, 1, 0)
            point_seat_mask_2 = torch.where(coors[0:1, :, 0] < 0.1, 1, 0)
            point_seat_mask_3 = torch.where(coors[0:1, :, 2] > -0.05, 1, 0)
            point_seat_mask_4 = torch.where(coors[0:1, :, 2] < 0.05, 1, 0)
            point_seat_mask = point_seat_mask_1 * point_seat_mask_2 * point_seat_mask_3 * point_seat_mask_4
            batch_size = point_seat_mask.shape[0]
            y_seat = []
            for i in range(batch_size):
                idx = torch.nonzero(point_seat_mask[i])[:, 0]
                y_seat.append(max(coors[i][idx][:, 1]).unsqueeze(0))
            y_seat = torch.cat(y_seat, dim=0).unsqueeze(1).unsqueeze(1)
            # divide according to y-position (divide from seat position)
            mask_2 = torch.where(coors[0:1, :, 1:2] > y_seat - 0.2, 1, 0)
            real_mask_split = [mask_1 * mask_2, mask_1 * (1 - mask_2), (1 - mask_1) * mask_2,
                               (1 - mask_1) * (1 - mask_2)]
            real_mask_split = torch.cat(real_mask_split, dim=-1)

            [recon_coors_wo, _, _], _ = transform_result(uv, mask, recon_coors[0:1])
            [recon_colors_wo, _, _], _ = transform_result(uv, mask, recon_colors[0:1])
            [recon_normals_wo, _, _], _ = transform_result(uv, mask, recon_normals[0:1])

            [coors_wo, _, _], _ = transform_result(uv, mask, coors[0:1])
            [colors_wo, _, _], _ = transform_result(uv, mask, colors[0:1])
            [normals_wo, _, _], _ = transform_result(uv, mask, normals[0:1])

            [real_coors_wo, _, _], _ = transform_result(uv, real_mask_split, coors[0:1])
            [real_colors_wo, _, _], _ = transform_result(uv, real_mask_split, colors[0:1])
            [real_normals_wo, _, _], _ = transform_result(uv, real_mask_split, normals[0:1])

            recon_imgs_wo = [recon_coors_wo.transpose(2, 0, 1), recon_colors_wo.transpose(2, 0, 1),
                             recon_normals_wo.transpose(2, 0, 1)]
            imgs_wo = [coors_wo.transpose(2, 0, 1), colors_wo.transpose(2, 0, 1), normals_wo.transpose(2, 0, 1)]
            real_imgs_wo = [real_coors_wo.transpose(2, 0, 1), real_colors_wo.transpose(2, 0, 1),
                            real_normals_wo.transpose(2, 0, 1)]

        if mode != 'train':
            viser.vis_loss_val(epoch_num=epoch, losses=losses_epoch)
        else:
            viser.vis_loss_train_epoch(epoch_num=epoch, losses=losses_epoch)
            viser.tc_vis_input_image_warp_result_train(epoch, image=imgs_wo, memo='predict_mask')
            viser.tc_vis_input_image_warp_result_train(epoch, image=recon_imgs_wo, memo='recon')
            viser.tc_vis_input_image_warp_result_train(epoch, image=real_imgs_wo, memo='real')
        return losses_epoch

    def optim_one_epoch(mode, epoch, dataloader, model, optimizers, viser):
        loss_priors, loss_smooths, loss_colors, loss_normals, loss_coors = [], [], [], [], []
        optimizer_encoder, optimizer_masker, optimizer_mapper, optimizer_generator = optimizers
        if mode == 'train':
            model = model.train()
        else:
            model = model.eval()

        for batch_id, data in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
            global batch_idx
            # print(batch_idx)
            if mode == 'train':
                optimizer_encoder.zero_grad()
                optimizer_masker.zero_grad()
                optimizer_mapper.zero_grad()
                optimizer_generator.zero_grad()
            # prapare data.
            coors, colors, normals, path = data
            coors, colors, normals = coors.float().cuda(), colors.float().cuda(), normals.float().cuda()
            # forward.
            losses = model(coors, colors, normals)
            if len(args.gpus) > 1:
                for idx, loss in enumerate(losses):
                    losses[idx] = losses[idx].sum() / len(args.gpus)
            # compute loss.
            i = 0
            for idx in range(len(args.adjust_weights)):
                if (epoch >= args.adjust_weights[idx]):
                    i = idx
            args.weights[i][3] = 0
            loss_total = losses[0] * args.weights[i][0] + losses[1] * args.weights[i][1] + losses[2] * args.weights[i][
                2] + losses[3] * args.weights[i][3] + losses[4] * args.weights[i][4]
            # losses = [losses[0] * args.weights[i][0], losses[1] * args.weights[i][1], losses[2] * args.weights[i][2], losses[3] * args.weights[i][3], losses[4] * args.weights[i][4]]
            # backward.
            if mode == 'train':
                loss_total.backward()
                optimizer_encoder.step()
                optimizer_masker.step()
                optimizer_mapper.step()
                optimizer_generator.step()
                batch_idx += 1
            # save and vis [weighted] loss.
            for idx, loss in enumerate(losses):
                losses[idx] = losses[idx] * args.weights[i][idx]

            loss_priors.append(losses[0].item())
            loss_smooths.append(losses[1].item())
            loss_coors.append(losses[2].item())
            loss_colors.append(0)
            loss_normals.append(losses[4].item())
        # if(batch_idx % args.num_vis == 0):
        # 	viser.vis_loss_train(batch_index=batch_idx, losses=losses)
        losses_epoch = [np.mean(loss_priors), np.mean(loss_smooths), np.mean(loss_coors), np.mean(loss_colors),
                        np.mean(loss_normals)]

        with torch.no_grad():
            from alignChairTexture import transform_result
            from torch.nn import Sigmoid, Softmax
            shape_code, coeffs = model.module.net['shape_encoder'](coors[0:1], colors[0:1])
            uv = model.module.net['uv_mapper'](coors[0:1], shape_code)
            print('uv: ', uv.min(), uv.max())
            pred_normal, mask = model.module.net['shape_masker'](coors[0:1], shape_code)

            second_mask = Sigmoid()(torch.sum(pred_normal * normals[0:1], dim=-1, keepdim=True))
            mask = [second_mask * mask, second_mask * (1 - mask), (1 - second_mask) * mask,
                    (1 - second_mask) * (1 - mask)]
            mask = torch.cat(mask, dim=-1)

            sep_pcs = []
            for i in range(args.num_generator):
                resized_coeff = coeffs[i].reshape(1, args.num_N, 9)
                basis = model.module.net['basis_generator_' + str(i)](uv)
                sep_pcs.append(torch.bmm(basis, resized_coeff))
                if i == 0:
                    recon_pcs = sep_pcs[i] * mask[:, :, i:i + 1]
                else:
                    recon_pcs += sep_pcs[i] * mask[:, :, i:i + 1]
            recon_coors, recon_colors, recon_normals = recon_pcs.chunk(3, dim=2)

            max_y_cord, _ = torch.max(coors[0:1, :, 1], dim=1)
            y_ = coors[0:1, :, 1] - max_y_cord.unsqueeze(1) - 0.05
            points_ = torch.cat([coors[0:1, :, 0:1], y_.unsqueeze(2), coors[0:1, :, 2:]], dim=2)
            # divide according to front and back (shown in normal direction)
            mask_1 = torch.where(torch.sum(points_ * normals[0:1], dim=-1, keepdim=True) < 0, 1, 0)
            point_seat_mask_1 = torch.where(coors[0:1, :, 0] > 0, 1, 0)
            point_seat_mask_2 = torch.where(coors[0:1, :, 0] < 0.1, 1, 0)
            point_seat_mask_3 = torch.where(coors[0:1, :, 2] > -0.05, 1, 0)
            point_seat_mask_4 = torch.where(coors[0:1, :, 2] < 0.05, 1, 0)
            point_seat_mask = point_seat_mask_1 * point_seat_mask_2 * point_seat_mask_3 * point_seat_mask_4
            batch_size = point_seat_mask.shape[0]
            y_seat = []
            for i in range(batch_size):
                idx = torch.nonzero(point_seat_mask[i])[:, 0]
                y_seat.append(max(coors[i][idx][:, 1]).unsqueeze(0))
            y_seat = torch.cat(y_seat, dim=0).unsqueeze(1).unsqueeze(1)
            # divide according to y-position (divide from seat position)
            mask_2 = torch.where(coors[0:1, :, 1:2] > y_seat - 0.2, 1, 0)
            real_mask_split = [mask_1 * mask_2, mask_1 * (1 - mask_2), (1 - mask_1) * mask_2,
                               (1 - mask_1) * (1 - mask_2)]
            real_mask_split = torch.cat(real_mask_split, dim=-1)

            [recon_coors_wo, _, _], _ = transform_result(uv, mask, recon_coors[0:1])
            [recon_colors_wo, _, _], _ = transform_result(uv, mask, recon_colors[0:1])
            [recon_normals_wo, _, _], _ = transform_result(uv, mask, recon_normals[0:1])

            [coors_wo, _, _], _ = transform_result(uv, mask, coors[0:1])
            [colors_wo, _, _], _ = transform_result(uv, mask, colors[0:1])
            [normals_wo, _, _], _ = transform_result(uv, mask, normals[0:1])

            [real_coors_wo, _, _], _ = transform_result(uv, real_mask_split, coors[0:1])
            [real_colors_wo, _, _], _ = transform_result(uv, real_mask_split, colors[0:1])
            [real_normals_wo, _, _], _ = transform_result(uv, real_mask_split, normals[0:1])

            recon_imgs_wo = [recon_coors_wo.transpose(2, 0, 1), recon_colors_wo.transpose(2, 0, 1),
                             recon_normals_wo.transpose(2, 0, 1)]
            imgs_wo = [coors_wo.transpose(2, 0, 1), colors_wo.transpose(2, 0, 1), normals_wo.transpose(2, 0, 1)]
            real_imgs_wo = [real_coors_wo.transpose(2, 0, 1), real_colors_wo.transpose(2, 0, 1),
                            real_normals_wo.transpose(2, 0, 1)]

        if mode != 'train':
            viser.vis_loss_val(epoch_num=epoch, losses=losses_epoch)
        else:
            viser.vis_loss_train_epoch(epoch_num=epoch, losses=losses_epoch)
            viser.tc_vis_input_image_warp_result_train(epoch, image=imgs_wo, memo='predict_mask')
            viser.tc_vis_input_image_warp_result_train(epoch, image=recon_imgs_wo, memo='recon')
            viser.tc_vis_input_image_warp_result_train(epoch, image=real_imgs_wo, memo='real')
        return losses_epoch

    '''CREATE DIR'''
    experiment_dir = Path(args.out_path)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.model + args.comments)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    viser = visualizer(log_dir)

    '''BACKUP CODE'''
    zip_name = os.path.join(experiment_dir) + "/code.zip"
    filelist = []
    for root, dirs, files in os.walk(ROOT_DIR):
        for name in files:
            filelist.append(os.path.join(root, name))
    zip_code = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
    for tar in filelist:
        arcname = tar[len(ROOT_DIR):]
        zip_code.write(tar, arcname)
    zip_code.close()

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('Parameters: ')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    TRAIN_DATASET = DataLoader(root=args.data_path, npoint=args.num_train_points, split='train')
    VAL_DATASET = DataLoader(root=args.data_path, npoint=args.num_train_points, split='val')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, pin_memory=True, drop_last=True)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10,
                                                pin_memory=True, drop_last=True)

    '''MODEL LOADING'''
    shutil.copy('./alignment_minxian/model.py', str(experiment_dir))
    model = auv_net(args)
    if len(args.gpus) > 1:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=args.gpus[0])
    # model = model.module
    else:
        model = model.cuda()

    manual_seed = random.randint(1, 10000)
    torch.cuda.manual_seed_all(manual_seed)

    '''OPTIMIZER SETTING'''
    optimizer_encoder, optimizer_masker, optimizer_mapper, optimizer_generator = model.module.create_optimizer()
    optimizers = [optimizer_encoder, optimizer_masker, optimizer_mapper, optimizer_generator]
    if args.continue_train:
        checkpoint = torch.load(os.path.join(args.out_path, args.model + args.comments, 'checkpoints', 'optimizer.pth'))
        optimizer_encoder.load_state_dict(checkpoint['encoder'])
        optimizer_masker.load_state_dict(checkpoint['masker'])
        optimizer_mapper.load_state_dict(checkpoint['mapper'])
        optimizer_generator.load_state_dict(checkpoint['generator'])
        trained_epoch = checkpoint['epoch']
    else:
        trained_epoch = 0
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=40, gamma=0.1)
    scheduler_masker = torch.optim.lr_scheduler.StepLR(optimizer_masker, step_size=40, gamma=0.1)
    scheduler_mapper = torch.optim.lr_scheduler.StepLR(optimizer_mapper, step_size=40, gamma=0.1)
    scheduler_generator = torch.optim.lr_scheduler.StepLR(optimizer_generator, step_size=40, gamma=0.1)

    '''TRANING'''
    global_epoch = 0
    loss_total_best = 1000
    logger.info('---Start training---')
    for epoch in range(args.epoch):
        if epoch < trained_epoch:
            log_string('\nEpoch %d (%d/%s): trained.' % (global_epoch + 1, epoch + 1, args.epoch))
            global_epoch += 1
            continue
        # training.
        log_string('\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        if epoch >= 10:
            print(1)
        if args.is_optimization:
            losses = optim_one_epoch('train', epoch, trainDataLoader, model, optimizers, viser, )
        else:
            losses = one_epoch('train', epoch, trainDataLoader, model, optimizers, viser, )

        scheduler_encoder.step()
        scheduler_masker.step()
        scheduler_mapper.step()
        scheduler_generator.step()
        # validation.
        """
        with torch.no_grad():
            if args.is_optimization:
                losses = optim_one_epoch('val', epoch, trainDataLoader, model, optimizers, viser, )
            else:
                losses = one_epoch('val', epoch, valDataLoader, model, optimizers, viser)
            loss_total = np.sum(losses) / len(args.gpus)
            if (loss_total <= loss_total_best):
                loss_total_best = loss_total
                logger.info('Saving [~BSET~] model at epoch - ' + str(epoch + 1))
                model.module.save('best')
            if (epoch % args.num_save_cks == 0):
                logger.info('Saving least model at epoch - ' + str(epoch + 1))
                model.module.save('least')
                # save optimizer.
                torch.save({'epoch': epoch,
                            'encoder': optimizer_encoder.state_dict(),
                            'masker': optimizer_masker.state_dict(),
                            'mapper': optimizer_mapper.state_dict(),
                            'generator': optimizer_generator.state_dict()},
                           os.path.join(args.out_path + args.model + args.comments, 'checkpoints', 'optimizer.pth'))
        """
        global_epoch += 1
    logger.info('---End of training---')


if __name__ == '__main__':
    args = parse_args()
    main(args)

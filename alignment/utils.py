import importlib
import numpy as np
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter


class visualizer(object):
    """Visualize losses."""

    def __init__(self, output_dir):
        self.writer = SummaryWriter(str(output_dir))

    def tex_split(self, img_tensor):
        img_tensor = img_tensor[0, ::].squeeze()
        # img_tensor = torch.chunk(img_tensor, 2, dim=2)
        # img_tensor = torch.cat([img_tensor[0], img_tensor[1]], dim=2)
        img_tensor = img_tensor.permute(1, 2, 0)
        return img_tensor

    def vis_loss_train(self, batch_index, losses):
        self.writer.add_scalar('train/loss', (losses[0].sum() + losses[1] + losses[2] + losses[3] + losses[4]),
                               batch_index)
        self.writer.add_scalar('train/loss_prior', losses[0], batch_index)
        self.writer.add_scalar('train/loss_smooth', losses[1], batch_index)
        self.writer.add_scalar('train/loss_coor', losses[2], batch_index)
        self.writer.add_scalar('train/loss_color', losses[3], batch_index)
        self.writer.add_scalar('train/loss_normal', losses[4], batch_index)

    def vis_loss_train_epoch(self, epoch_num, losses):
        self.writer.add_scalar('train/loss', np.sum(losses), epoch_num)
        self.writer.add_scalar('train/loss_prior', losses[0], epoch_num)
        self.writer.add_scalar('train/loss_smooth', losses[1], epoch_num)
        self.writer.add_scalar('train/loss_coor', losses[2], epoch_num)
        self.writer.add_scalar('train/loss_color', losses[3], epoch_num)
        self.writer.add_scalar('train/loss_normal', losses[4], epoch_num)

    def vis_loss_val(self, epoch_num, losses):
        self.writer.add_scalar('validation/loss', np.sum(losses), epoch_num)
        self.writer.add_scalar('validation/loss_prior', losses[0], epoch_num)
        self.writer.add_scalar('validation/loss_smooth', losses[1], epoch_num)
        self.writer.add_scalar('validation/loss_coor', losses[2], epoch_num)
        self.writer.add_scalar('validation/loss_color', losses[3], epoch_num)
        self.writer.add_scalar('validation/loss_normal', losses[4], epoch_num)

    def tc_vis_loss_train_epoch(self, epoch_num, losses):
        self.writer.add_scalar('train/loss', np.sum(losses), epoch_num)
        self.writer.add_scalar('train/loss_recon', losses[0], epoch_num)

    def tc_vis_loss_val_epoch(self, epoch_num, losses):
        self.writer.add_scalar('validation/loss', np.sum(losses), epoch_num)
        self.writer.add_scalar('validation/loss_recon', losses[0], epoch_num)

    def tc_ae_vis_loss_train_epoch(self, epoch_num, losses):
        self.writer.add_scalar('train/loss', losses[0] + losses[1] + losses[2], epoch_num)
        self.writer.add_scalar('train/loss_recon', losses[0], epoch_num)
        self.writer.add_scalar('train/loss_mask', losses[1], epoch_num)
        self.writer.add_scalar('train/loss_code', losses[2], epoch_num)

    def tc_ae_vis_loss_val_epoch(self, epoch_num, losses):
        self.writer.add_scalar('validation/loss', losses[0] + losses[1] + losses[2], epoch_num)
        self.writer.add_scalar('validation/loss_recon', losses[0], epoch_num)
        self.writer.add_scalar('validation/loss_mask', losses[1], epoch_num)
        self.writer.add_scalar('validation/loss_code', losses[2], epoch_num)

    def tc_nocs_vis_loss_train_epoch(self, epoch_num, losses):
        self.writer.add_scalar('train/loss', losses[0] + losses[1] + losses[2] + losses[3] + losses[4], epoch_num)
        self.writer.add_scalar('train/loss_nocs', losses[0], epoch_num)
        self.writer.add_scalar('train/loss_norm', losses[1], epoch_num)
        self.writer.add_scalar('train/loss_code', losses[2], epoch_num)
        self.writer.add_scalar('train/loss_tex', losses[3], epoch_num)
        self.writer.add_scalar('train/loss_mask', losses[4], epoch_num)

    def tc_nocs_vis_loss_val_epoch(self, epoch_num, losses):
        self.writer.add_scalar('validation/loss', losses[0] + losses[1] + losses[2] + losses[3] + losses[4], epoch_num)
        self.writer.add_scalar('validation/loss_nocs', losses[0], epoch_num)
        self.writer.add_scalar('validation/loss_norm', losses[1], epoch_num)
        self.writer.add_scalar('validation/loss_code', losses[2], epoch_num)
        self.writer.add_scalar('validation/loss_tex', losses[3], epoch_num)
        self.writer.add_scalar('validation/loss_mask', losses[4], epoch_num)

    def vis_GAN_loss(self, batch_idx, losses):
        self.writer.add_scalar('G/total', losses['G_Recon'] + losses['G_ImgD_GAN'] + losses['G_TexD_GAN'] + losses['G_TexD_coor'], batch_idx)
        self.writer.add_scalar('G/G_Recon', losses['G_Recon'], batch_idx)
        self.writer.add_scalar('G/G_ImgD_GAN', losses['G_ImgD_GAN'], batch_idx)
        self.writer.add_scalar('G/G_TexD_GAN', losses['G_TexD_GAN'], batch_idx)
        self.writer.add_scalar('G/G_TexD_coor', losses['G_TexD_coor'], batch_idx)
        self.writer.add_scalar('D/total', losses['D_ImgD_GAN'] + losses['D_TexD_GAN'] + losses['D_TexD_coor'],
                               batch_idx)
        self.writer.add_scalar('D/D_ImgD_GAN', losses['D_ImgD_GAN'], batch_idx)
        self.writer.add_scalar('D/D_TexD_GAN', losses['D_TexD_GAN'], batch_idx)
        self.writer.add_scalar('D/D_TexD_coor', losses['D_TexD_coor'], batch_idx)

    def vis_GAN_res(self, batch_idx, res):
        self.writer.add_image('1.Input', img_tensor=res['input'][0][0:3, :, :], global_step=batch_idx,
                              dataformats='CHW')
        # self.writer.add_image('2.MapNet/Norm', img_tensor=res['norm'][0], global_step=batch_idx, dataformats='HWC')
        # self.writer.add_image('2.MapNet/NOCS', img_tensor=res['nocs'][0], global_step=batch_idx, dataformats='HWC')
        self.writer.add_image('2.MapNet/PrtColor', img_tensor=res['prtColor'][0][0:3, :, :], global_step=batch_idx,
                              dataformats='CHW')
        self.writer.add_image('3.GenNet/ComColor',
                              img_tensor=(res['comColor'][0][0:3, :, :] * res['shapeFeature'][0][0:3, :, :]),
                              global_step=batch_idx, dataformats='CHW')
        self.writer.add_image('3.GenNet/GtColor', img_tensor=(res['gt_texture'][0][0:3, :, :]), global_step=batch_idx,
                              dataformats='CHW')

    def vis_GAN_res_val(self, batch_idx, res):
        self.writer.add_image('4.Input', img_tensor=res['input'][0][0:3, :, :], global_step=batch_idx,
                              dataformats='CHW')
        # self.writer.add_image('2.MapNet/Norm', img_tensor=res['norm'][0], global_step=batch_idx, dataformats='HWC')
        # self.writer.add_image('2.MapNet/NOCS', img_tensor=res['nocs'][0], global_step=batch_idx, dataformats='HWC')
        self.writer.add_image('5.MapNet/PrtColor', img_tensor=res['prtColor'][0][0:3, :, :], global_step=batch_idx,
                              dataformats='CHW')
        self.writer.add_image('6.GenNet/ComColor',
                              img_tensor=(res['comColor'][0][0:3, :, :] * res['shapeFeature'][0][0:3, :, :]),
                              global_step=batch_idx, dataformats='CHW')
        self.writer.add_image('6.GenNet/GtColor', img_tensor=(res['gt_texture'][0][0:3, :, :]), global_step=batch_idx,
                              dataformats='CHW')

    def tc_vis_input_image(self, batch_index, image):
        self.writer.add_image('3.Input/image', img_tensor=image[0][0:3, :, :], global_step=batch_index,
                              dataformats='CHW')

    # self.writer.add_image('1.Input/mask', img_tensor=image[0][3:4,:,:], global_step=batch_index, dataformats='CHW')

    def tc_vis_resut_image(self, batch_index, resized_tex, recon_tex):
        resized_tex = self.tex_split(resized_tex)
        recon_tex = self.tex_split(recon_tex)
        self.writer.add_image('4.Output/recon', img_tensor=recon_tex, global_step=batch_index, dataformats='HWC')
        self.writer.add_image('4.Output/gt', img_tensor=resized_tex, global_step=batch_index, dataformats='HWC')

    def tc_vis_input_image_train(self, batch_index, image):
        self.writer.add_image('3.Input/image', img_tensor=image[0][0:3, :, :], global_step=batch_index,
                              dataformats='CHW')

    def tc_vis_resut_image_train(self, batch_index, resized_tex, recon_tex):
        resized_tex = self.tex_split(resized_tex)
        recon_tex = self.tex_split(recon_tex)
        self.writer.add_image('4.Output/recon', img_tensor=recon_tex, global_step=batch_index, dataformats='HWC')
        self.writer.add_image('4.Output/gt', img_tensor=resized_tex, global_step=batch_index, dataformats='HWC')

    def tc_vis_input_image_nocs(self, batch_index, image):
        self.writer.add_image('3.Input/image', img_tensor=image[0][0:3, :, :], global_step=batch_index,
                              dataformats='CHW')

    def tc_vis_resut_image_nocs(self, batch_index, recon_tex, resized_tex, recon_nocs, gt_nocs, recon_norm, gt_norm):
        resized_tex = self.tex_split(resized_tex)
        recon_tex = self.tex_split(recon_tex)
        self.writer.add_image('4.Output/tex_recon', img_tensor=recon_tex, global_step=batch_index, dataformats='HWC')
        self.writer.add_image('4.Output/tex_gt', img_tensor=resized_tex, global_step=batch_index, dataformats='HWC')
        self.writer.add_image('4.Output/nocs_recon', img_tensor=recon_nocs[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')
        self.writer.add_image('4.Output/nocs_gt', img_tensor=gt_nocs[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')
        self.writer.add_image('4.Output/norm_recon', img_tensor=recon_norm[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')
        self.writer.add_image('4.Output/norm_gt', img_tensor=gt_norm[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')

    def tc_vis_input_image_nocs_train(self, batch_index, image):
        self.writer.add_image('1.Train-Input/image', img_tensor=image[0][0:3, :, :], global_step=batch_index,
                              dataformats='CHW')

    def tc_vis_resut_image_nocs_train(self, batch_index, recon_tex, resized_tex, recon_nocs, gt_nocs, recon_norm,
                                      gt_norm):
        resized_tex = self.tex_split(resized_tex)
        recon_tex = self.tex_split(recon_tex)
        self.writer.add_image('2.Train-Output/tex_recon', img_tensor=recon_tex, global_step=batch_index,
                              dataformats='HWC')
        self.writer.add_image('2.Train-Output/tex_gt', img_tensor=resized_tex, global_step=batch_index,
                              dataformats='HWC')
        self.writer.add_image('2.Train-Output/nocs_recon', img_tensor=recon_nocs[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')
        self.writer.add_image('2.Train-Output/nocs_gt', img_tensor=gt_nocs[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')
        self.writer.add_image('2.Train-Output/norm_recon', img_tensor=recon_norm[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')
        self.writer.add_image('2.Train-Output/norm_gt', img_tensor=gt_norm[0, :, :, :], global_step=batch_index,
                              dataformats='CHW')

    def tc_vis_input_image_warp_result_train(self, batch_index, image, memo):
        self.writer.add_image('1.Train-Input/coors-' + memo, img_tensor=image[0][0:3, :, :], global_step=batch_index, dataformats='CHW')
        self.writer.add_image('1.Train-Input/colors-' + memo, img_tensor=image[1][0:3, :, :], global_step=batch_index, dataformats='CHW')
        self.writer.add_image('1.Train-Input/normals-' + memo, img_tensor=image[2][0:3, :, :], global_step=batch_index, dataformats='CHW')



class saver(object):
    """Save checkpoints."""

    def __init__(self):
        self.mode = 0

    def save(self, model, epoch, dir, mgpu, loss, mode):
        shape_encoder, shape_masker, uv_mapper, basis_generators = model
        if mode == 'BEST':
            pref = 'best'
        else:
            pref = str(epoch)
        # shape encoder.
        savepath = str(dir) + '/' + pref + '_encoder.pth'
        state = {
            'epoch': epoch + 1,
            'total loss': loss,
            'model_state_dict': (shape_encoder.module.state_dict() if mgpu else shape_encoder.state_dict())
        }
        torch.save(state, savepath)
        # shape masker.
        savepath = str(dir) + '/' + pref + '_masker.pth'
        state = {
            'epoch': epoch + 1,
            'total loss': loss,
            'model_state_dict': (shape_masker.module.state_dict() if mgpu else shape_masker.state_dict())
        }
        torch.save(state, savepath)
        # uv mapper.
        savepath = str(dir) + '/' + pref + '_mapper.pth'
        state = {
            'epoch': epoch + 1,
            'total loss': loss,
            'model_state_dict': (uv_mapper.module.state_dict() if mgpu else uv_mapper.state_dict())
        }
        torch.save(state, savepath)
        # generator.
        for idx in range(len(basis_generators)):
            basis_generator = basis_generators[idx]
            savepath = str(dir) + '/' + pref + '_generator' + str(idx) + '.pth'
            state = {
                'epoch': epoch + 1,
                'total loss': loss,
                'model_state_dict': (basis_generator.module.state_dict() if mgpu else basis_generator.state_dict())
            }
            torch.save(state, savepath)


def load_network(net, label, args, mode):
    if 'tc-net' in args.model:
        tp = 'tc_net'
    else:
        tp = 'auv_net'
    save_filename = '%s_%s.pth' % (mode, label)
    save_dir = os.path.join('../../data/auv_output/', tp, args.model + args.comments, 'checkpoints')
    save_path = os.path.join(save_dir, save_filename)
    if not os.path.exists(save_path):
        print('not find model :' + save_path + ', do not load model!')
        return net
    weights = torch.load(save_path)
    try:
        net.load_state_dict(weights)
    except KeyError:
        print('key error, not load!')
    except RuntimeError as err:
        print(err)
        net.load_state_dict(weights, strict=False)
        print('loaded with strict=False')
    return net


def save_network(net, label, epoch, args):
    if 'tc-net' in args.model:
        tp = 'tc_net'
    elif 'pipeline' in args.model:
        tp = 'pipeline'
    else:
        tp = 'auv_net'
    save_filename = '%s_%s.pth' % (epoch, label)
    save_path = os.path.join('../../data/auv_output/', tp, args.model + args.comments, 'checkpoints', save_filename)
    torch.save(net.state_dict(), save_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from util.cp_dataset import CPDataset, CPDataLoader
from util.utils import load_checkpoint, save_checkpoint
from gmm import GMM
from gmm_gma import RAFTGMA
from gen import ResUnetGenerator
from loss import GicLoss, VGGLoss

from tensorboardX import SummaryWriter
from util.visualization import board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--stage", default="GMM")

    # parser.add_argument("--name", default="TOM")
    # parser.add_argument("--stage", default="TOM")

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="../../data/cpvton_plus")
    parser.add_argument("--datamode", default="train")

    parser.add_argument("--data_list", default="train_pairs.txt")

    parser.add_argument("--fine_width", type=int, default=384)
    parser.add_argument("--fine_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=200)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    iter_start_time = time.time()

    for step in range(opt.keep_step + opt.decay_step):

        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        im_pose = inputs['pose_image'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        agnostic = inputs['agnostic'].cuda()

        agnostic_gmm = F.interpolate(agnostic, size=(256, 192), mode='nearest')
        cm_gmm = F.interpolate(cm, size=(256, 192), mode='nearest')

        grid = model(agnostic_gmm, cm_gmm)

        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)

        visuals = [[im_h, shape, im_pose], [c, warped_cloth, im_c], [warped_mask, (warped_cloth+im)*0.5, im]]

        Lgic = gicloss(grid)
        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])
        Lwarp = criterionL1(warped_cloth, im_c)
        loss = Lwarp + 40 * Lgic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('loss', loss.item(), step+1)
            board.add_scalar('40*Lgic', (40*Lgic).item(), step+1)
            board.add_scalar('Lwarp', Lwarp.item(), step+1)
            t = time.time() - iter_start_time
            iter_start_time = time.time()
            print('step: %8d, time: %.3f, loss: %4f, (40*Lgic): %.8f, Lwarp: %.6f' %
                  (step+1, t, loss.item(), (40*Lgic).item(), Lwarp.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    model_gmm = GMM(opt, inputA_nc=22, inputB_nc=1)
    model_gmm.eval()
    model_gmm.cuda()
    load_checkpoint(model_gmm, 'checkpoints/GMM/step_200000.pth')

    iter_start_time = time.time()

    for step in range(opt.keep_step + opt.decay_step):
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()

        #### gmm
        agnostic_gmm = F.interpolate(agnostic, size=(256, 192), mode='nearest')
        cm_gmm = F.interpolate(cm, size=(256, 192), mode='nearest')
        grid = model_gmm(agnostic_gmm, cm_gmm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)

        #### gen
        gen_inputs = torch.cat([agnostic, warped_cloth, warped_mask], 1)
        outputs = model(gen_inputs)
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        """visuals = [[im_h, shape, im_pose], [c, cm*2-1, m_composite*2-1], [p_rendered, p_tryon, im]]"""  # CP-VTON
        visuals = [[im_h, shape, im_pose], [warped_cloth, pcm*2-1, m_composite*2-1], [p_rendered, p_tryon, im]]  # CP-VTON+

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        # loss_mask = criterionMask(m_composite, cm)  # CP-VTON
        loss_mask = criterionMask(m_composite, pcm)  # CP-VTON+
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            iter_start_time = time.time()
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step+1, t, loss.item(), loss_l1.item(), loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def main():
    torch.backends.cudnn.benchmark = True
    opt = get_opt()
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)

    os.makedirs(opt.tensorboard_dir, exist_ok=True)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    if opt.stage == 'GMM':
        model = GMM(opt, inputA_nc=22, inputB_nc=1)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)

    if opt.stage == 'TOM':
        model = ResUnetGenerator(input_nc=26, output_nc=4, num_downs=5)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()

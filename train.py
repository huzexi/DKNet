import argparse
import sys
from datetime import datetime
from os import path
from time import time
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from component import History, logger, set_log_file
from component.config import CustomConfig as config, config_to_json
from component.dataset import get_dataset, BaseDataset
from component.network import create_model
from component.util import get_dir, get_git_hash
from component.util.evaluate import mse2psnr, Evaluator
from component.util.torchsummary import summary_to_txt


def main():
    parser = argparse.ArgumentParser(description='AngularSR PyTorch implementation.')
    parser.add_argument('--name', help='Name of this run, used as dir name when generating files', type=str, default='')
    parser.add_argument('--gpuid', help='GPU to be used', type=str, default='0')
    parser.add_argument('--continue_name', help="Name of the task to be continued", type=str, default='')
    parser.add_argument('--continue_epoch', help="Starting epoch", type=int, default=-1)

    # Arguments parsing
    args = parser.parse_args()

    # Initialization
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Continue training
    to_continue = args.continue_name != ''
    if to_continue:
        task_name = args.continue_name
    else:
        task_name = datetime.now().strftime('%Y%m%d.%H%M%S') + '_' + args.name

    # Model
    model = create_model(config)
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr)

    # Losses
    criterion = nn.MSELoss()
    criterion_name = 'MSEloss'

    # Setting device
    use_cuda = torch.cuda.is_available() and args.gpuid != "-1"
    device = torch.device("cuda:%s" % args.gpuid if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    model = model.to(device)
    criterion = criterion.to(device)

    # Checkpoint, Tensorboard and CSV History
    cwd = get_dir(config.dir_workdir, 'train', task_name)
    pth_ckp = path.join(cwd, "%05d.pt")
    tb_writer = SummaryWriter(path.join(cwd))
    his_train = History(path.join(cwd, 'history_train.csv'), [criterion_name, 'PSNR', 'time'])
    his_valid = History(path.join(cwd, 'history_valid.csv'), [criterion_name, 'PSNR', 'SSIM', 'time'])
    if not to_continue:
        his_train.init()
        his_valid.init()
    set_log_file(path.join(cwd, "all.log"))
    logger.info("Task name: %s" % task_name)
    logger.info("Git commit version: %s" % get_git_hash())

    # Dataset for train
    train_ds = get_dataset(config.dataset, config, BaseDataset.MODE_TRAIN)
    train_loader = DataLoader(dataset=train_ds,
                              num_workers=config.train_workers,
                              batch_size=config.train_batch_sz,
                              pin_memory=True,
                              shuffle=False)
    # Dataset for validation
    if config.valid:
        valid_ds = get_dataset(config.dataset, config, BaseDataset.MODE_VALID)
        valid_loader = DataLoader(dataset=valid_ds,
                                  num_workers=1,
                                  batch_size=1,
                                  pin_memory=True,
                                  shuffle=False)
        if config.valid_slice_n > 1:
            slice_cfg = {
                "slice_n": config.valid_slice_n,
                "slice_offset": config.valid_slice_offset,
                "scale": config.scale
            }
        else:
            slice_cfg = None
        validor = Evaluator(loader=valid_loader, model=model, shave=config.shave, criterion=criterion,
                            to_monitor=True, mp=config.valid_mp, slice_cfg=slice_cfg)
        logger.info("Pre-testing the model.")
        validor.test_once()

    # Iterations
    best = {criterion_name: sys.float_info.max}
    if to_continue:
        ep_start = args.continue_epoch
        model.load_state_dict(torch.load(pth_ckp % ep_start, map_location=device))
    else:
        ep_start = 0

    # Saving configuration
    config_to_json(config, path.join(cwd, 'Config.json'))
    summary_to_txt(model, config, device, path.join(cwd, 'Summary.txt'))

    for ep in range(ep_start + 1, config.train_ep + 1):
        model.train()
        train_ckp_ep = config.get_train_ckp_ep(ep)

        pbar = tqdm(total=config.train_ep_iter,
                    desc="%s-Epoch[%05d]" % (task_name, ep))
        loss_train = np.zeros((len(train_loader), 2))
        t_train = time()
        for i, (x, gt) in enumerate(train_loader):
            # Forward propagation
            x, gt = x.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, gt)
            loss_train[i, 0] = loss.item()
            loss_train[i, 1] = mse2psnr(loss_train[i, 0])
            # Backward propagation
            loss.backward()
            optimizer.step()
            # Update
            pbar.update()

        # After an epoch
        t_train = time() - t_train
        t_train = t_train * 1000 / len(train_loader)
        loss_train = loss_train.mean(0)
        pbar.close()
        logger.info('Epoch[%05d] Train: %s: %.8f, PSNR: %.4f, Time: %dms/step' %
                    (ep, criterion_name, loss_train[0], loss_train[1], t_train))
        tb_writer.add_scalar(f"Train/{criterion_name}", loss_train[0], ep)
        tb_writer.add_scalar(f"Train/PSNR", loss_train[1], ep)
        tb_writer.add_scalar(f"Train/speed (ms/step)", t_train, ep)
        his_train.write(ep,
                        ["%.8f" % loss_train[0], '%.2f' % loss_train[1],
                         "%d" % t_train])  # Primary, display losses, time

        # Checkpoint
        if (ep - ep_start) % train_ckp_ep == 0:
            # Valid
            if config.valid:
                t_valid = time()

                model.eval()
                validor.set_model(model)
                res_lst = validor.run()

                t_valid = time() - t_valid
                t_valid = t_valid * 1000

                res_crit = np.array([res['criterion'] for res in res_lst]).mean(0)
                res_psnr = np.array([res['psnr'] for res in res_lst]).mean(0)
                res_ssim = np.array([res['ssim'] for res in res_lst]).mean(0)
                is_best = res_crit < best[criterion_name]
                best = {criterion_name: res_crit, 'epoch': ep} if is_best else best

                logger.info("Epoch[%05d] Valid: %s: %.8f, PSNR: %.2f, SSIM: %.4f. Best epoch is %d epochs ago." %
                            (ep, criterion_name, res_crit, res_psnr, res_ssim, ep - best['epoch']))
                tb_writer.add_scalar(f"Valid/{criterion_name}", res_crit, ep)
                tb_writer.add_scalar(f"Valid/PSNR", res_psnr, ep)
                tb_writer.add_scalar(f"Valid/SSIM", res_ssim, ep)
                tb_writer.add_scalar(f"Valid/speed (ms/image)", t_valid, ep)
                his_valid.write(ep, ["%.8f" % res_crit, "%.2f" % res_psnr, "%.4f" % res_ssim,
                                     "%d" % t_valid])

                if not is_best and config.valid_save_best and (ep - ep_start) % config.valid_force_save_ep != 0:
                    continue  # Skip saving not best model

            # Save checkpoint
            torch.save(model.state_dict(), pth_ckp % ep)
            logger.info("Model saved: %s" % (pth_ckp % ep))

    tb_writer.flush()


if __name__ == '__main__':
    main()

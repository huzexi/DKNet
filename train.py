import argparse
import sys
from datetime import datetime
from os import path
from time import time

import numpy as np
import pyprind
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    crit_pri = nn.MSELoss()
    name_pri = 'MSEloss'  # Primary loss

    # Setting device
    use_cuda = torch.cuda.is_available() and args.gpuid != "-1"
    device = torch.device("cuda:%s" % args.gpuid if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    model = model.to(device)
    crit_pri = crit_pri.to(device)

    # Checkpoint and history
    cwd = get_dir(config.dir_ckp, task_name)
    pth_ckp = path.join(cwd, "%05d.pt")
    his_train = History(path.join(cwd, 'history_train.csv'), [name_pri, 'PSNR', 'time'])
    his_valid = History(path.join(cwd, 'history_valid.csv'), [name_pri, 'PSNR', 'SSIM', 'time'])
    if not to_continue:
        his_train.init()
        his_valid.init()
    set_log_file(path.join(cwd, "all.log"))
    logger.info("Task name: %s" % task_name)
    logger.info("Git commit version: %s" % get_git_hash())

    # Dataset
    ## Train
    train_ds = get_dataset(config.dataset, config, BaseDataset.MODE_TRAIN)
    train_loader = DataLoader(dataset=train_ds,
                              num_workers=5,
                              batch_size=config.train_batch_sz,
                              pin_memory=True,
                              shuffle=False)
    # Valid
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
        validor = Evaluator(loader=valid_loader, model=model, shave=config.shave, criterion=crit_pri,
                            to_monitor=True, mp=config.valid_mp, slice_cfg=slice_cfg)
        logger.info("Pre-testing the model.")
        validor.test_once()

    # Iterations
    best = {name_pri: sys.float_info.max}
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

        bar = pyprind.ProgBar(iterations=config.train_ep_iter,
                              title="Training %s Epoch[%05d]" % (task_name, ep),
                              width=80)
        loss_train = np.zeros((len(train_loader), 2))
        t_train = time()
        for i, (x, gt) in enumerate(train_loader):
            x, gt = x.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = crit_pri(pred, gt)
            loss_train[i, 0] = loss.item()
            loss_train[i, 1] = mse2psnr(loss_train[i, 0])

            loss.backward()
            optimizer.step()

            bar.update()

        t_train = time() - t_train
        t_train = t_train * 1000 / len(train_loader)
        loss_train = loss_train.mean(0)
        bar.stop()
        logger.info('Epoch[%05d]: %s: %.8f, PSNR: %.4f, Time: %dms/step' %
                    (ep, name_pri, loss_train[0], loss_train[1], t_train))
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

                loss_crit = np.array([res['criterion'] for res in res_lst]).mean(0)
                loss_psnr = np.array([res['psnr'] for res in res_lst]).mean(0)
                loss_ssim = np.array([res['ssim'] for res in res_lst]).mean(0)
                is_best = loss_crit < best[name_pri]
                best = {name_pri: loss_crit, 'epoch': ep} if is_best else best

                logger.info("--------------------------------------------------------------------------------------------")
                logger.info("Validation of Epoch[%05d]: %s: %.8f, PSNR: %.2f, SSIM: %.4f. Best epoch is %d epochs ago." %
                            (ep, name_pri, loss_crit, loss_psnr, loss_ssim, ep - best['epoch']))
                logger.info("--------------------------------------------------------------------------------------------")
                his_valid.write(ep, ["%.8f" % loss_crit, "%.2f" % loss_psnr, "%.4f" % loss_ssim,
                                     "%d" % t_valid])

                if not is_best and config.valid_save_best and (ep - ep_start) % config.valid_force_save_ep != 0:
                    continue  # Skip saving not best model

            # Save checkpoint
            torch.save(model.state_dict(), pth_ckp % ep)
            logger.info("Model saved: %s" % (pth_ckp % ep))


if __name__ == '__main__':
    main()

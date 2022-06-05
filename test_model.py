import argparse
from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader

from component.config import CustomConfig
from component.dataset import BaseDataset, get_dataset
from component.network import create_model
from component.util import get_dir, full2scene_name
from component.util.evaluate import Evaluator


def main():
    # Parse args
    parser = argparse.ArgumentParser(description='SpatialSR PyTorch test.')
    parser.add_argument('--model', help="Model path.", type=str, required=True)
    parser.add_argument('--bgr', help="Render BGR PNG.", action='store_true')
    parser.add_argument('--gpuid', help="ID of the gpu to be used", type=str, default='0')
    args = parser.parse_args()
    config = CustomConfig

    # Model and device
    model = create_model(config)
    use_cuda = torch.cuda.is_available() and args.gpuid != "-1"
    device = torch.device("cuda:%s" % args.gpuid if use_cuda else "cpu")
    if use_cuda:
        model.to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Load, predict and evaluate
    datasets = ['Yeung']
    for ds in datasets:
        test_ds = get_dataset(ds, config, BaseDataset.MODE_TEST)
        test_loader = DataLoader(dataset=test_ds,
                                 num_workers=1,
                                 batch_size=1,  # Different samples have different size, cannot test in sample > 1
                                 pin_memory=True,
                                 shuffle=False
                                 )
        names = test_ds.names
        cwd = get_dir(config.dir_tmp, 'test_model')

        if config.test_slice_n > 1:
            slice_cfg = {
                "slice_n": config.test_slice_n,
                "slice_offset": config.test_slice_offset,
                "scale": config.scale
            }
        else:
            slice_cfg = None
        
        # Test
        with torch.no_grad():
            model(torch.zeros([1, 3, 8, 8, 32, 32], device=device))

        evaluator = Evaluator(loader=test_loader, model=model, shave=config.shave, criterion=None,
                              to_monitor=True, render_bgr=cwd if args.bgr else None, slice_cfg=slice_cfg,
                              mp=config.test_mp)
        res_lst = evaluator.run()

        psnr_lst = [res['psnr'] for res in res_lst]
        ssim_lst = [res['ssim'] for res in res_lst]
        psnr_score = np.mean(psnr_lst)
        ssim_score = np.mean(ssim_lst)
        print("Final: PSNR: %.2f, SSIM: %.4f" % (float(psnr_score), float(ssim_score)))

        # Individual sample result
        save_pth = path.join(cwd, '%s.csv' % ds)
        with open(save_pth, 'w') as f:
            f.write(','.join(['sample', 'psnr', 'ssim']) + '\n')
            for i in range(len(psnr_lst)):
                f.write(','.join(
                    [full2scene_name(names[i]), "%.2f" % psnr_lst[i], "%.4f" % ssim_lst[i]]
                ) + '\n')


if __name__ == "__main__":
    main()

# DKNet

This repo is the implementation of Decomposition Kernel Network (DKNet) for light field super-resolution.
> **Texture-Enhanced Light Field Super-Resolution With Spatio-Angular Decomposition Kernels**   
> Zexi Hu; Xiaoming Chen; Henry Wing Fung Yeung; Yuk Ying Chung; Zhibo Chen  
> IEEE Transactions on Instrumentation and Measurement (2022)   
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/9721149)  [[arXiv]](https://arxiv.org/abs/2111.04069)

### Citation
Please cite us if it is helpful for your work:

```
@article{hu2022texture,
  title={Texture-Enhanced Light Field Super-Resolution With Spatio-Angular Decomposition Kernels},
  author={Hu, Zexi and Chen, Xiaoming and Yeung, Henry Wing Fung and Chung, Yuk Ying and Chen, Zhibo},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={71},
  pages={1--16},
  year={2022},
  publisher={IEEE}
}
```


## How to run
### Dependencies
Most dependencies have been listed in `requirements.txt`, simply run the following command to install them:
```bash
pip install -r requirements.txt
```
The code is tested with Python 3.8/3.10 and Ubuntu 20.04/22.04.

### Data preparation
This step is to loading the huge LF images into HDF5 files, which simplifies the process and accelerate the speed when training.

Modify the paths in  `component/dataset/LytroDataset.py` to the corresponding locations, e.g. `Stanford` and `SIGASIA16`, and run the following command:
```bash
python prepare_dataset.py
```

The command uses `component/dataset/YeungDataset.py`, which inherits from `LytroDataset` and specify samples for the training and testing set as in [Yeung's work](https://ieeexplore.ieee.org/abstract/document/8561240/).

Please be reminded that this step will read the LF images into HDF5 files which might be big. The `Yeung` dataset is nearly 40GB.

### Testing

Run the following command will infer the given model on the prepared dataset calculating PSNR and SSIM and producing images in `workdir/test` directory.

```bash
python test.py --model /path/to/weight.pt --bgr
```

Refer to the [Release](https://github.com/huzexi/DKNet/releases) page for the pretrained weight files.

### Training

Run the following command to launch a training session named `train1`. 

```bash
python train.py --name train1
```

A folder will be created in `workdir/train` containing all the files generated during the training, including weight files, logs, configuration, etc. You can also run the following command to monitor the training process using `tensorboard`.

```bash
tensorboard --logdir workdir/train
```

## Configuration
Readers are suggested to check the following configuration files for customisation.
- `component/config/DefaultConfig.py` includes most of the configuration variables as default, `component/config/CustomConfig.py` overrides the default and is used when running.
- `component/network/DKNet/DKNetConfig.py` includes the configuration of DKNet.

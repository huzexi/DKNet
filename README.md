# DKNet
## ⚠️ This is a preliminary version. We will update more information soon.

This repo is the implementation of Decomposition Kernel Network (DKNet) for light field super-resolution.
> **Texture-Enhanced Light Field Super-Resolution With Spatio-Angular Decomposition Kernels**   
> Zexi Hu; Xiaoming Chen; Henry Wing Fung Yeung; Yuk Ying Chung; Zhibo Chen  
> IEEE Transactions on Instrumentation and Measurement (2022)   
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/9721149)  [[arXiv]](https://arxiv.org/abs/2111.04069)



# How to run
### Dependencies
Most dependencies have been listed in `requirements.txt`, simply run the following command to install them:
```bash
pip install -r requirements.txt
```
The code is tested with Python 3.8 and 3.10 and Ubuntu 22.04.

### Data preparation
Modify the paths in `component/dataset/LytroDataset.py` to the corresponding locations, and run the following command:
```bash
python prepare_dataset.py
```

Please be reminded that this step will read the LF images into hdf5 files which might be big. The `Yeung` dataset is nearly 40GB.

### Testing the model
Run the following command will infer the given model calculating PSNR and SSIM and producing images in `tmp/test_model` directory.

```bash
python test_model.py --model /path/to/weight.pt --bgr
```

The best model (18 Gamma kernels) weight file can be downloaded [here](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/zehu6197_uni_sydney_edu_au/EU0TJGIhY9ZDo8-cta-8ISoB_5L2lXNbFGmqIpzc2WtuwQ?e=8vxCWM).
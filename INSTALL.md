# Installation Guide

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n gamma_quant_inertial python==3.13.2
conda activate gamma_quant_inertial
```

Install PyTorch distribution (CUDA 12.4):

```
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

Install other packages:

```
pip install -r requirements.txt
```

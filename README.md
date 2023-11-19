# Live Version of the Avalanche COVID XRAY Classifier

## Setup

### Install Avalanche

As per the website: https://avalanche.continualai.org/getting-started/how-to-install

````pip install git+https://github.com/ContinualAI/avalanche.git
Install pytorch and torchvision```

As per the pytorch website: https://pytorch.org/

pip3 install torch torchvision torchaudio

Make sure that Tensorboard is installed with pip install tensorboard To run tensorboard make sure that the tensorboard logger is used. Next run tensorboard with --logdir argument and then the folder path to the tensorboard logs.

`tensorboard --logdir tb_data/`

## Python Version

Python version used 3.9.2

Requirements as per requirements.txt

````

avalanche-lib
matplotlib==3.5.0
numpy==1.21.4

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

#torch==1.10.1+cu113
#torchvision==0.11.2+cu113
#torchaudio==0.10.1+cu113
tqdm==4.62.3
typing-extensions
psutil
tensorboard
scikit-learn
pytorchcv
quadprog
gdown
pycocotools
discord_notify
seaborn
torchviz
python-dotenv
Flask
firebase-admin

```

After installing Avalanche

Might need to install CUDA versions of torch and torchvision. Instructiosn found at https://pytorch.org/get-started/locally/

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

Otherwise a similar error like this will appear

NVIDIA GeForce RTX 3080 Ti with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA GeForce RTX 3080 Ti GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

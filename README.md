# CL MLP update

Official PyTorch implementation.<br>
MLP with attention for classification<br>
Shiqin Zeng<br>


## Requirements

Make sure the installation of miniconda3 and the conda environment can be set up using these commands:

```.bash
conda create -n py310_env python=3.10
conda activate py310_env
conda install pytorch torchvision torchaudio -c pytorch
```

## Matlab (2024 version) 
Switch MATLAB to Use the Compatible Environment: Point MATLAB to the Python executable in the new environment (change to your path):
```.bash
pyenv('Version', 'C:\Users\Shiqin\miniconda3\envs\py310_env\python.EXE');
```

## Test the pretrained model
Download the pre-trained model and put it in the [matlab_scripts](matlab_scripts), run the [main.m](matlab_scripts/main.m) to test the model.







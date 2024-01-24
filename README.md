# Project Name

The Application of Three Reinforcement Learning Methods in Inventory Management

## Installation Dependencies

Make sure your environment meets the following requirements:

- Ubuntu 18.04
- Python 3.8

### Install CUDA 11.3 and cuDNN 8

Follow the official documentation to install [CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-downloads) and [cuDNN 8](https://developer.nvidia.com/cudnn).

### Install NVCC

Ensure that NVCC (NVIDIA CUDA Compiler) is correctly installed. It is usually included with the CUDA Toolkit.

### Install PyTorch 1.11.0
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
### Install TensorFlow 2.4.0
    pip install tensorflow==2.4.0
### Install TensorFlow Probability 0.12.0
    pip install tensorflow-probability==0.12.0
### Install tqdm 4.65.0
    pip install tqdm==4.65.0
### Install Keras 2.13.1
    pip install keras==2.13.1

## Usage

### Running the Jupyter Notebooks
    pip install jupyter
    jupyter notebook

## Directory Structure
```
.
├── exp_results/
├── agent.py
├── environment.py
├── game.py
├── qlearning.py
├── A2C-5-1000.ipynb
├── A2C-1000-5.ipynb
├── qlearning-5-1000.ipynb
├── qlearning-1000-5.ipynb
├── multi-A2C-5-1000.ipynb
├── multi-A2C-1000-5.ipynb
├── README.md
└── requirements.txt

# SenseLLM
This code repository presents our unified large model for multi-task wireless sensing, SenseLLM.

## 0. Prerequisite

SenseLLM is implemented with [Python 3.8](https://www.python.org/downloads/release/python-380/) and [PyTorch 2.0.1](https://pytorch.org/get-started/previous-versions/). We manage the development environment using [Conda](https://docs.conda.io/en/latest/). Execute the following commands to configure the development environment.

- Create a conda environment called `SenseLLM` based on Python 3.8, and activate the environment.

```bash
conda create -n myenv python=3.8
conda activate myenv
```
- Install PyTorch, as well as other required packages.
    ```bash
    pip3 install torch
    ```
    ```bash
    pip3 install numpy scipy tensorboard tqdm matplotlib torchvision pytorch_fid
    ```

For more details about the environment configuration, refer to the `requirements.txt` file.


## 1. Train and Test
All detailed model code of SenseLLM is located in the Model directory. To reproduce this process, please follow the steps below.
- Fall Detection: Fill in the dataset path and run the following command in the terminal:
```bash
CUDA_VISIBLE_DEVICES=1 python Train_Test_comprehensive.py --task 1 --gpu 1
```
- Gesture Recognition: Fill in the dataset path and run the following command in the terminal:
```bash
CUDA_VISIBLE_DEVICES=2 python Train_Test_comprehensive.py --task 2 --gpu 2
```
- Gait Recognition: Fill in the dataset path and run the following command in the terminal:
```bash
CUDA_VISIBLE_DEVICES=3 python Train_Test_comprehensive.py --task 3 --gpu 3
```
To reduce the reproduction difficulty and computational cost, we only provide the lightweight GPT-2 as the LLM backbone in the code. For other backbones such as Qwen or LLaMA, please download them from their official websites.
## 2. Dataset of SenseLLM
To successfully run the code, you need to set the dataset path in the code as follows:


Data Path of Fall Detection:
```bash
 ./denoisemat
```
Data Path of Gesture Recognition:
```bash
 ./20250708
```
Data Path of Gait Recognition:
```bash
 ./Gait
```

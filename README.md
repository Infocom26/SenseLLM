# SenseLLM

**SenseLLM** is a **unified large language model (LLM) framework** for **multi-task wireless sensing**.  
It leverages the **reasoning, representation learning, and cross-task generalization** abilities of LLMs to handle diverse sensing tasks within a single architecture.

## 🌟 Highlights
- **Unified Foundation Model** – A single LLM backbone for multiple wireless sensing tasks, breaking the one-model-per-task limitation.
- **Cross-Task Generalization** – Learns shared and task-specific features, enabling seamless adaptation to new sensing scenarios.
- **Task-Aware Representation Learning** – Dynamically adapts sensing features based on task requirements.
- **Scalable to Stronger LLMs** – Default GPT-2 for reproducibility, but can be replaced with Qwen, LLaMA, or other large-scale pre-trained models.

---

## 0. Prerequisite

SenseLLM is implemented with [Python 3.8](https://www.python.org/downloads/release/python-380/) and [PyTorch 2.0.1](https://pytorch.org/get-started/previous-versions/).  
We recommend using [Conda](https://docs.conda.io/en/latest/) to manage the development environment.

- Create a conda environment and activate it:
```bash
conda create -n SenseLLM python=3.8
conda activate SenseLLM
```
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


Note: For ease of reproduction and reduced computational cost, we provide lightweight GPT-2 as the default LLM backbone.
To fully exploit SenseLLM’s capabilities, replace GPT-2 with more powerful backbones such as Qwen or LLaMA.
## 2. Dataset of SenseLLM
To successfully run the code, you need to set the dataset path in the code as follows:


Data Path of Fall Detection:
```bash
 ./denoisemat
```
Data Path of Gesture Recognition:
```bash
 ./Gesture
```
Data Path of Gait Recognition:
```bash
 ./Gait
```

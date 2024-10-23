# 🫧 Capsule Vision Challenge 2024 : Solution

Welcome to my submission for the **[Capsule Vision Challenge 2024: Multi-Class Abnormality Classification for Video Capsule Endoscopy](https://arxiv.org/abs/2408.04940)**. This repository contains the code, models, and instructions for training, evaluating and creation of submission file.

---

## 🌟 Project Overview

This solution is designed to solve automatic classification of abnormalities captured in VCE video frames using deep learning techniques. The model leverages **[EfficientViT](https://github.com/mit-han-lab/efficientvit)** and code is adapted from its [official implementation](https://github.com/mit-han-lab/efficientvit).

---

## 📊 Solution Report

Authorea link : 

---

## Installation
```
conda create -n efficientvit python=3.10
conda activate efficientvit
cd efficientvit
pip install -r requirements.txt
```
---

## Training

Download the pre-trained model (imagenet) model from [here](https://www.icloud.com/iclouddrive/0aa43CSXmSwJITAuxD8Zrswng#l2-r224) and put it in `efficientvit/assets/checkpoints` folder.

To train the model, follow the instructions below:

```bash
python simple_train.py --config config.yaml # from efficientvit directory
```

The dataset format should be in following structure:

```bash
imagenet/
├── train/
│       ├── Angioectasia/
│       │       └── image1.jpg
│       │       └── image2.jpg
│       ├── Bleeding/
│       │       └── ...
│       ├── Normal/
│       └── ...
└── val/s
        ├── Angioectasia/
        │       └── image111.jpg
        │       └── image211.jpg
        ├── Bleeding/
        │   └── ...
        ├── Normal/
        └── ...
```


This will start training using the default configuration file. `train_config.yaml` can be modified for custom settings like learning rate, epochs, batch size, and more.

---

## Evaluation 

### ▶ Evaluation on training and validation dataset on trained model

The instructions for evaluating the model on training and validation datasets given in
[***efficientvit / eval_model.ipynb***](efficientvit/eval_model.ipynb) notebook.

### ▶ Creation of submission file
Run the below command to create the submission file.
```bash
python create_submission.py --test_folder <test_folder_path> --output_file <output_file_path> --model_path <model_path> --num_classes 10
```
---

## 📬 Trained model :
The trained model used for submission can be downloaded from the following link:
https://www.icloud.com/iclouddrive/0a0Zh2XgogcD8DWtx0FYshqsw#EfficientViT-L2-submission-capsvision2024-Team-eAI


## 📬 Contact

- **Email**: girin.iitm@gmail.com

---

### 📄 License

This project is licensed under the MIT License.

---

### ✨ Acknowledgments

A big thanks to challenge organizers for providing this opportunity. 🙏


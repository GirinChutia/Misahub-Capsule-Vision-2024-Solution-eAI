# 🫧 Capsule Vision Challenge 2024 : Solution

Welcome to my submission for the **[Capsule Vision Challenge 2024: Multi-Class Abnormality Classification for Video Capsule Endoscopy](https://arxiv.org/abs/2408.04940)**. This repository contains the code, models, and instructions for training, evaluating and creation of submission file.

---

## 🌟 Project Overview

This solution is designed to solve automatic classification of abnormalities captured in VCE video frames using deep learning techniques. The model leverages **[EfficientViT](https://github.com/mit-han-lab/efficientvit)** and code is adapted from its [official implementation](https://github.com/mit-han-lab/efficientvit).

## 📊 Solution Report

Arxiv link : 

## Installation
```
conda create -n efficientvit python=3.10
conda activate efficientvit
cd efficientvit
pip install -r requirements.txt
```
---

## Training

To train the model, follow the instructions below:

```bash
python simple_train.py --config config.yaml # from efficientvit directory
```

The dataset format should be in following structure:

```bash
imagenet/
├── train/
│       ├── Angioectasia/
│       │   └── ...
│       ├── Bleeding/
│       │   └── ...
│       ├── Normal/
│       └── ...
└── val/s
        ├── Angioectasia/
        │   └── ...
        ├── Bleeding/
        │   └── ...
        ├── Normal/
        └── ...
```


This will start training using the default configuration file. You can modify `train_config.yaml` for your custom settings like learning rate, epochs, batch size, and more.

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


## 📬 Contact

- **Author**: Girin Chutia
- **Email**: girin.iitm@gmail.com

---

### 📄 License

This project is licensed under the MIT License.

---

### ✨ Acknowledgments

A big thanks to challenge organizers for providing this opportunity. 🙏


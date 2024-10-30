import argparse
import glob
import os
import torch
from simple_train import load_model
from eval_model import get_image_paths, single_image_inference, infer_folder, evaluate_model
import pandas as pd
from tqdm import tqdm

# ======== Dataset Folder Format ========

# dataset_folder
# ├── training
# │   ├── Angioectasia
# │   │   ├── KID
# │   │   │    ├── image121.png
# │   │   │    ├── ....
# │   │   ├── KVASIR
# │   │   │    ├── image33.png
# │   │   │    ├── ....
# │   ├── Bleeding
# │   │   ├── KID
# │   │   │    ├── image11.png
# │   │   │    ├── ....
# │   │   ├── KVASIR
# │   │   │    ├── image121.png
# │   │   │    ├── ....
# │   ├── ...
# ├── validation
# │   ├── Angioectasia
# │   │   ├── KID
# │   │   │    ├── image11.png
# │   │   │    ├── ....
# │   │   ├── KVASIR
# │   │   │    ├── image15.png
# │   │   │    ├── ....
# │   ├── Bleeding
# │   │   ├── KID
# │   │   │    ├── image131.png
# │   │   │    ├── ....
# │   │   ├── KVASIR
# │   │   │    ├── image152.png
# │   │   │    ├── ....

# =============================

def infer_train_val(data):
    """
    Perform inference on the training and validation datasets.

    Args:
        data (list): A list of tuples containing the full path, image path, ground truth label and dataset name of all images in the dataset folder

    Returns:
        pd.DataFrame: A DataFrame containing the image paths, dataset names and predicted probabilities for each class
    """
    image_paths_list = []
    labels = []
    pred_confs = []
    pred_classes = []
    pred_class_names = []
    all_prob_pred = []
    dataset_names = []

    class_names = [
        "Angioectasia",
        "Bleeding",
        "Erosion",
        "Erythema",
        "Foreign Body",
        "Lymphangiectasia",
        "Normal",
        "Polyp",
        "Ulcer",
        "Worms",
    ]

    # Iterate over all images in the dataset
    for dp in tqdm(data, total=len(data)):
        fullpath = dp[0]
        imagepath = dp[1]
        gt_label = dp[2]
        dataset = dp[3]
        # Perform single image inference
        prob_preds, pred_conf, pred_class, pred_class_name = single_image_inference(model, fullpath, device)
        # Append results to the lists
        image_paths_list.append(imagepath)
        labels.append(gt_label)
        pred_confs.append(pred_conf)
        pred_classes.append(pred_class)
        pred_class_names.append(pred_class_name)
        all_prob_pred.append(prob_preds)
        dataset_names.append(dataset)

    # Create a DataFrame from the lists
    df = pd.DataFrame(
        {
            "image_path": image_paths_list,
            "Dataset": dataset_names,
            # "true_label": labels,
            # "predicted_class": pred_class_names,
        }
    )

    # Add columns for each class
    for i, class_name in enumerate(class_names):
        df[f"{class_name}"] = [prob_pred[i] for prob_pred in all_prob_pred]

    return df

def get_train_val_data(dataset_folder):
    """
    Get a list of tuples containing the full path and image path of all images in the dataset folder

    Args:
        dataset_folder (str): The path to the dataset folder

    Returns:
        list: A list of tuples containing the full path and image path of all images in the dataset folder
    """
    _data = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".jpg"):
                full_path = os.path.join(root, file)
                # Get the class label and dataset name from the folder structure
                parts = full_path.split(os.sep)
                classlabel = parts[-3]
                dataset = parts[-2]
                # Create the relative path to the image
                dpath = [parts[-4], parts[-3], parts[-2], parts[-1]]
                dpath = '\\'.join(dpath)
                # Add the full path and relative path to the list
                _data.append([full_path, dpath, classlabel, dataset])
    return _data

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate EfficientViT model on capsule endoscopy images")

    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to root folder of training & validatio dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the best model checkpoint")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the classification task")
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_arguments()
    
    # Load the model
    model = load_model(num_classes=args.num_classes)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    
    # Move the model to the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    
    print(f"Model path : {args.model_path}")
    print("Model Loaded ..")
    
    # training dataset inference
    train_folder = os.path.join(args.dataset_folder, "training")
    val_folder = os.path.join(args.dataset_folder, "validation")
    
    print(f"Inferring Training folder: {train_folder}")
    train_data = get_train_val_data(train_folder)
    print(f"Inferring Validation folder: {val_folder}")
    val_data = get_train_val_data(val_folder)
    
    # Create dataframes for training and validation datasets
    train_df = infer_train_val(train_data)
    val_df = infer_train_val(val_data)
    
    # Save the results to Excel files
    train_df.to_excel("eAI_predicted_train_dataset.xlsx",
                      index=False)
    val_df.to_excel("eAI_predicted_val_dataset.xlsx", 
                    index=False)
    
    
    
    

import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
import glob, os
from torchvision.datasets.folder import default_loader
import torch
from simple_train import load_model, build_valid_transform
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def find_jpg_images(base_path):
    """
    Find all JPEG images in a folder and its subfolders.
    
    Args:
        base_path (str): The root directory to start searching from.
    
    Returns:
        list: A list of full paths to all JPEG images found.
    """
    # Define the pattern to match JPEG files
    pattern = os.path.join(base_path, "**", "*.jpg")

    try:
        # Use glob to find all matching files
        jpeg_files = glob.glob(pattern, recursive=True)

        return jpeg_files

    except PermissionError:
        print("Permission denied for some directories.")
        return []


def get_image_paths(TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER):
    """
    Get image paths from folders and store them in a dictionary with their labels.

    Args:
        TRAIN_FOLDER (str): The path to the training folder.
        VAL_FOLDER (str): The path to the validation folder.
        TEST_FOLDER (str): The path to the test folder.

    Returns:
        tuple: A tuple containing the following:
            - train_paths (list): A list of dictionaries containing the image path and label.
            - val_paths (list): A list of dictionaries containing the image path and label.
            - test_image_paths (list): A list of image paths in the test folder.
    """
    train_folders = [path for path in glob.glob(os.path.join(TRAIN_FOLDER, "*")) if os.path.isdir(path)]
    val_folders = [path for path in glob.glob(os.path.join(VAL_FOLDER, "*")) if os.path.isdir(path)]
    test_image_paths = glob.glob(os.path.join(TEST_FOLDER, "*.jpg"))

    print(f"Number of training folders: {len(train_folders)}")
    print(f"Number of validation folders: {len(val_folders)}")
    print(f"Number of test images: {len(test_image_paths)}")

    train_paths = []
    val_paths = []

    train_image_count = {}
    val_image_count = {}

    for folder in train_folders:
        jpg_images = find_jpg_images(folder)
        # print(folder, len(jpg_images))
        # Count the number of images in the folder
        train_image_count[os.path.basename(folder)] = len(jpg_images)
        # Iterate over the images in the folder and add their path and label to the list
        for image in jpg_images:
            train_paths.append({"image_path": image, "label": os.path.basename(folder)})

    for folder in val_folders:
        jpg_images = find_jpg_images(folder)
        # print(folder, len(jpg_images))
        # Count the number of images in the folder
        train_image_count[os.path.basename(folder)] = len(jpg_images)
        # Iterate over the images in the folder and add their path and label to the list
        for image in jpg_images:
            val_paths.append({"image_path": image, "label": os.path.basename(folder)})

    print(f"Number of training images: {len(train_paths)}")
    print(f"Number of validation images: {len(val_paths)}")

    return train_paths, val_paths, test_image_paths


def single_image_inference(model, image_path, device):
    """
    Perform inference on a single image.

    Args:
        model: A PyTorch model.
        image_path: The path to the image file.
        device: The device to use for inference (e.g. 'cuda' or 'cpu').

    Returns:
        A tuple containing the following:
            - prob_preds: A list of the predicted probabilities for each class.
            - pred_conf: The confidence of the predicted class.
            - pred_class: The index of the predicted class.
            - pred_class_name: The name of the predicted class.
    """

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
    transform = build_valid_transform(224)
    image = default_loader(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    y_hat = torch.softmax(output, dim=1)
    y_hat = y_hat.detach().cpu()
    prob_preds = list(y_hat.numpy()[0])
    prob_preds = [round(x, 4) for x in prob_preds]
    pred_conf, pred_class = torch.max(y_hat, dim=1)
    pred_conf = pred_conf.item()
    pred_class = pred_class.item()
    pred_class_name = class_names[pred_class]
    return prob_preds, pred_conf, pred_class, pred_class_name


def infer_folder(model, image_paths, type="test", device="cuda"):
    """
    Perform inference on a folder of images.

    Args:
        model: A PyTorch model.
        image_paths: A list of dictionaries containing the path to the image and the true label.
        type: The type of inference to perform, i.e. 'train', 'val', or 'test'.
        device: The device to use for inference (e.g. 'cuda' or 'cpu').

    Returns:
        A Pandas DataFrame containing the results of the inference. The columns will vary depending on the type of inference.
    """
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

    # Initialize lists to store results
    image_paths_list = []
    labels = []
    pred_confs = []
    pred_classes = []
    pred_class_names = []
    all_prob_pred = []

    # Process images
    total_images = len(image_paths)
    for i in tqdm(range(total_images)):
        imp = image_paths[i]["image_path"] if type in ["train", "val"] else image_paths[i]
        label = image_paths[i]["label"] if type in ["train", "val"] else None

        # Perform single image inference
        prob_preds, pred_conf, pred_class, pred_class_name = single_image_inference(model, imp, device=device)

        # Append results to the lists
        image_paths_list.append(os.path.basename(imp))
        labels.append(label)
        pred_confs.append(pred_conf)
        pred_classes.append(pred_class)
        pred_class_names.append(pred_class_name)
        all_prob_pred.append(prob_preds)

    # Create the DataFrame
    if type in ["train", "val"]:
        df = pd.DataFrame(
            {
                "image_path": image_paths_list,
                "true_label": labels,
                "pred_class_id": pred_classes,
                "predicted_class": pred_class_names,
                "pred_confidence": pred_confs,
            }
        )

        # Expand probability predictions into separate columns for each class
        for i, class_name in enumerate(class_names):
            df[f"{class_name}"] = [prob_pred[i] for prob_pred in all_prob_pred]
    else:
        df = pd.DataFrame({"image_path": image_paths_list,})

        # Expand probability predictions into separate columns for each class
        for i, class_name in enumerate(class_names):
            df[f"{class_name}"] = [prob_pred[i] for prob_pred in all_prob_pred]

        df["predicted_class"] = pred_class_names

    return df


def evaluate_model(df):
    """
    Evaluate the performance of a model by calculating the classification report, mean AUC, and balanced accuracy.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the true labels and predicted labels and probabilities.

    Returns:
        dict: A dictionary containing the classification report, mean AUC, and balanced accuracy.
    """
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

    # Extract relevant data
    y_true = df["true_label"].values
    y_pred = df["predicted_class"].values
    y_pred_proba = df[[f"{cls}" for cls in class_names]].values

    # Calculate classification report
    class_report = classification_report(y_true, y_pred, target_names=class_names)

    # Calculate mean AUC
    try:
        auc_scores = roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
        mean_auc = round(np.mean(auc_scores), 2)
    except ValueError:
        mean_auc = None  # Set to None if calculation fails (e.g., single class present)

    # Calculate balanced accuracy
    balanced_acc = round(balanced_accuracy_score(y_true, y_pred), 3)

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()

    return {"classification_report": class_report, "mean_auc": mean_auc, "balanced_accuracy": balanced_acc}

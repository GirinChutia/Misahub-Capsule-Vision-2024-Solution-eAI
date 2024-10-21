import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,balanced_accuracy_score
import numpy as np

def calculate_mean_auc_confusion_matrix(y_true, y_pred_proba, class_names):
    """
    Calculate the Mean AUC, Confusion Matrix, and plot ROC curves for each class.

    Args:
    y_true (np.array): True labels (one-hot encoded) with shape (n_samples, n_classes).
    y_pred_proba (np.array): Predicted probabilities with shape (n_samples, n_classes).
    class_names (list): List of class names.

    Returns:
    mean_auc (float): The mean AUC across all classes.
    conf_matrix (np.array): The confusion matrix with shape (n_classes, n_classes).
    """
    # Calculate AUC for each class and plot ROC curve
    aucs = []
    num_classes = y_true.shape[1]
    
    plt.figure(figsize=(15, 10))
    
    for i in range(num_classes):
        # Calculate AUC
        auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        aucs.append(auc)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {auc:.2f})')
    
    mean_auc = np.mean(aucs)
    
    # Plot settings for ROC curves
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc='lower right')
    plt.show()
    
    # Convert one-hot encoded labels to class indices
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_true_labels, y_pred_labels)
    
    print(f"Mean AUC: {mean_auc:.2f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, class_names)
    
    return mean_auc, conf_matrix, aucs

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix as a heatmap.

    Args:
    conf_matrix (np.array): The confusion matrix to plot.
    class_names (list): List of class names corresponding to the confusion matrix.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def one_hot_encode(y, num_classes):
    """
    Convert integer class labels to one-hot encoded format.

    Args:
    y (np.array): Array of integer class labels (shape: n_samples).
    num_classes (int): Total number of classes.

    Returns:
    np.array: One-hot encoded array (shape: n_samples, num_classes).
    """
    # Initialize the one-hot encoded array with zeros
    one_hot = np.zeros((y.shape[0], num_classes))
    
    one_hot[np.arange(y.shape[0]), y] = 1
    
    return one_hot


def eval_model(model, val_dataset):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    img_paths = []
    y_all_probs = []
    allimage_paths = val_dataset.imgs
    indx = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device)
    with torch.no_grad():
        for x, y in tqdm(val_dataset, total=len(val_dataset)):
            y_hat = model(x.unsqueeze(0).to(device))
            y_hat = torch.softmax(y_hat, dim=1)
            y_hat = y_hat.cpu()
            prob_preds = list(y_hat.numpy()[0])
            pred_conf, pred_class = torch.max(y_hat, dim=1)
            pred_conf = pred_conf.item()
            pred_class = pred_class.item()
            y_all_probs.append(prob_preds)
            y_pred.append(pred_class)
            y_true.append(y)
            y_prob.append(pred_conf)
            img_paths.append(allimage_paths[indx][0])
            indx += 1
    
    results_df = pd.DataFrame({
        'Image_path': img_paths,
        'actual_class': y_true,
        'predicted_class': y_pred,
        'predicted_prob': y_prob,
        'pred_all_probs': y_all_probs
    })
    
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=val_dataset.classes, digits=3))
    
    # balanced accuracy
    print("Balanced Accuracy Score:")
    print(balanced_accuracy_score(y_true, y_pred))
    
    return results_df
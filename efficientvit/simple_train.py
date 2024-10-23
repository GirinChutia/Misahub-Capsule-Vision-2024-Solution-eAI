import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from efficientvit.cls_model_zoo import create_cls_model
from efficientvit.apps import setup
from efficientvit.models.nn import ConvLayer, LinearLayer, OpSequential

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os, math
import torchvision.transforms.functional as F
from efficientvit.models.utils import torch_random_choices

from efficientvit.cls_model_zoo import create_cls_model
from efficientvit.apps import setup
from efficientvit.models.nn import ConvLayer, LinearLayer, OpSequential
from efficientvit.models.nn.drop import apply_drop_func
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np
import glob
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassF1Score
from rich import print as rprint
from typing import Tuple, Dict, List
import yaml


def load_config(config_file):
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)


def get_interpolate(name: str) -> F.InterpolationMode:
    mapping = {
        "nearest": F.InterpolationMode.NEAREST,
        "bilinear": F.InterpolationMode.BILINEAR,
        "bicubic": F.InterpolationMode.BICUBIC,
        "box": F.InterpolationMode.BOX,
        "hamming": F.InterpolationMode.HAMMING,
        "lanczos": F.InterpolationMode.LANCZOS,
    }
    if name in mapping:
        return mapping[name]
    elif name == "random":
        return torch_random_choices(
            [
                F.InterpolationMode.NEAREST,
                F.InterpolationMode.BILINEAR,
                F.InterpolationMode.BICUBIC,
                F.InterpolationMode.BOX,
                F.InterpolationMode.HAMMING,
                F.InterpolationMode.LANCZOS,
            ],
        )
    else:
        raise NotImplementedError


def build_valid_transform(image_size) -> any:
    mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    crop_size = int(math.ceil(image_size / 1))
    return transforms.Compose(
        [
            transforms.Resize(crop_size, interpolation=get_interpolate("bicubic"),),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(**mean_std),
        ]
    )


def build_train_transform(image_size) -> any:
    mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    image_size = image_size
    train_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]

    use_post_aug = True
    post_aug = []

    if use_post_aug:
        random_erase = transforms.RandomErasing(p=0.1)
        random_brightness = transforms.RandomAdjustSharpness(1.2, p=0.1)
        random_rot = transforms.RandomRotation(degrees=(0, 180))
        random_contrast = transforms.RandomAutocontrast(p=0.1)

        post_aug.append(random_erase)
        post_aug.append(random_brightness)
        post_aug.append(random_rot)
        post_aug.append(random_contrast)

    train_transforms = [
        *train_transforms,
        transforms.ToTensor(),
        transforms.Normalize(**mean_std),
        *post_aug,
    ]
    return transforms.Compose(train_transforms)


def mixup(data, targets, alpha=0.2, p=0.5):
    if np.random.rand() < p:  # Apply mixup with probability p
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
        lam = np.random.beta(alpha, alpha, 1).item()
        mixed_data = lam * data + (1 - lam) * shuffled_data
        mixed_targets = lam * targets + (1 - lam) * shuffled_targets
        return mixed_data, mixed_targets
    else:  # Return original data and targets with probability 1-p
        return data, targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    model_save_path: str,
    num_epochs: int = 25,
    device: str = "cuda",
    batch_size: int = 16,
    learning_rate: float = 0.001,
    model_name: str = None,
    experiment_name: str = None,
    run_name: str = None,
    num_classes: int = 10,
    **kwargs,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains a PyTorch model for multiclass classification.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        optimizer: Optimizer (e.g., optim.Adam).
        model_save_path: Path to save the model.
        num_epochs: Number of epochs for training. Defaults to 25.
        device: Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        batch_size: Batch size for training. Defaults to 16.
        learning_rate: Learning rate for the optimizer. Defaults to 0.001.
        model_name: Name of the model. Defaults to None.
        experiment_name: Name of the experiment. Defaults to None.
        run_name: Name of the run. Defaults to None.
        num_classes: Number of classes. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        model: Trained model.
        history: Dictionary containing loss and accuracy history for training and validation.
    """
    model = model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}

    mlflow.set_experiment(experiment_name)

    best_valacc = 0
    best_valf1 = 0
    best_ckpt_paths = []

    os.makedirs(model_save_path, exist_ok=True)
    runcnt = len(glob.glob(model_save_path + "/*"))
    model_save_path = os.path.join(model_save_path, str(runcnt + 1))
    os.makedirs(model_save_path, exist_ok=False)
    print(f"Model save path: {model_save_path}")

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("model_save_path", model_save_path)

        if kwargs:
            for k, v in kwargs.items():
                mlflow.log_param(k, v)

        for epoch in range(num_epochs):

            # Training phase
            model.train()

            f1_score_train = MulticlassF1Score(num_classes=num_classes, average="micro")
            f1_score_val = MulticlassF1Score(num_classes=num_classes, average="micro")

            f1_score_train.to(device)
            f1_score_val.to(device)

            running_loss = 0.0
            running_corrects = 0
            val_loss = 0.0
            val_corrects = 0

            # Training loop
            for inputs, labels in tqdm(
                train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1} of {num_epochs}"
            ):

                labels_ohe = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

                # MIXUP AUG
                inputs, labels_ohe = mixup(inputs, labels_ohe, alpha=kwargs["mixup_alpha"], p=kwargs["mixup_p"])

                inputs, labels_ohe = inputs.to(device), labels_ohe.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                # outputs = out.softmax(dim=1)
                loss = criterion(outputs, labels_ohe)
                _, preds = torch.max(outputs, 1)  # eg pred : tensor([pred_class_no]) = tensor([8])

                # Backward pass
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Accumulate loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.cpu() == labels.cpu().data)

                f1_score_train(preds.to(device), labels.to(device))

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            trainf1 = float(f1_score_train.compute().item())

            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc.item())
            history["train_f1"].append(trainf1)

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc.item(), step=epoch)
            mlflow.log_metric("train_f1", trainf1, step=epoch)

            # Validation phase
            model.eval()

            with torch.no_grad():
                for inputs, labels in tqdm(
                    val_loader, total=len(val_loader), desc=f"Validating Epoch {epoch + 1} of {num_epochs}"
                ):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

                    f1_score_val(preds.to(device), labels.to(device))

            valf1 = float(f1_score_val.compute().item())

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)

            history["val_f1"].append(valf1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc.item())

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc.item(), step=epoch)
            mlflow.log_metric("val_f1", valf1, step=epoch)

            # Save model if val_acc is best
            if valf1 > best_valf1:
                rprint(f"New best validation f1: {valf1:.4f}, previous best: {best_valf1:.4f}")
                best_valf1 = valf1
                if len(best_ckpt_paths) > 0:
                    os.remove(best_ckpt_paths[-1])
                best_ckpt_path = os.path.join(model_save_path, f"{model_name}-bestvalf1-{epoch}-{valf1:.4f}.pt")
                torch.save(model.state_dict(), best_ckpt_path)
                best_ckpt_paths.append(best_ckpt_path)
                rprint(f"Saved model to {best_ckpt_path}")

            rprint(
                f"Epoch {epoch+1}/{num_epochs}:"
                f" Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f},"
                f" Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                f" Val f1: {valf1:.4f}, Train f1: {trainf1:.4f}"
            )

        # save last model
        torch.save(model.state_dict(), os.path.join(model_save_path, f"{model_name}-last.pt"))

    return model, history


def load_model(num_classes=10):
    """
    Load the model with pretrained weights and custom classifier head.

    Returns:
        model (nn.Module): The loaded model.
    """

    def count_parameters(model):
        """
        Count the number of trainable and non-trainable parameters in the model.

        Args:
            model (nn.Module): The model to count the parameters for.

        Returns:
            tuple: A tuple containing the number of trainable parameters and non-trainable parameters.
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params

    class ClassifierHead(OpSequential):
        """
        Custom classifier head for the model.
        """

        def __init__(self, in_channels, width_list, num_classes, dropout, norm, act_func, fid):
            """
            Initialize the classifier head.

            Args:
                in_channels (int): Number of input channels.
                width_list (list): Width of each layer.
                num_classes (int): Number of classes.
                dropout (float): Dropout rate.
                norm (str): Normalization method.
                act_func (str): Activation function.
                fid (str): Feature identifier.
            """
            super().__init__(
                [
                    ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
                    LinearLayer(width_list[1], num_classes, True, dropout, None, None),
                ]
            )
            self.feature_id = fid

        def forward(self, features):
            """
            Perform forward pass through the classifier head.

            Args:
                features (dict): Dictionary containing features.

            Returns:
                torch.Tensor: Output of the classifier head.
            """
            x = features[self.feature_id]
            return super().forward(x)

    # Load the model
    # config = setup.setup_exp_config(r'D:\Work\Challenges\Misahub-Capsule-Vision\models\efficientvit\configs\cls\imagenet\l2.yaml', recursive=True)
    model = create_cls_model("l2", True, dropout=0, num_classes=num_classes, weight_url="assets/checkpoints/l2-r224.pt")

    # Define the classifier head
    classifier_head = ClassifierHead(
        in_channels=512,
        width_list=[3072, 3200],
        num_classes=num_classes,
        dropout=0.0,
        norm="bn2d",
        act_func="gelu",
        fid="stage_final",
    )

    # Set the classifier head as the model's head
    model.head = classifier_head

    # Freeze the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Print the number of trainable and non-trainable parameters
    trainable_params, non_trainable_params = count_parameters(model)
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of non-trainable parameters: {non_trainable_params}")

    return model


if __name__ == "__main__":

    config_file = "config.yaml"  # Specify the path to your YAML file
    config = load_config(config_file)

    num_classes = config["num_classes"]
    model = load_model(num_classes=num_classes)

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    use_sampler = config["use_sampler"]
    use_class_weights = config["use_class_weights"]
    sampler_power = config["sampler_power"]

    experiment_name = config["experiment_name"]
    run_name = config["run_name"]
    optimizer_name = config["optimizer_name"]

    data_dir = config["data_dir"]
    model_save_path = config["model_save_path"]
    model_name = config["model_name"]
    mixup_alpha = config["mixup_alpha"]
    mixup_p = config["mixup_p"]

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_name == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    valid_transform = build_valid_transform(224)
    train_transform = build_train_transform(224)

    train_dataset = ImageFolder(os.path.join(data_dir, "train"), train_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, "val"), valid_transform)

    if use_sampler:
        class_counts = np.bincount(train_dataset.targets)
        pt_weights = (1.0 / class_counts) ** sampler_power
        pt_weights /= pt_weights.sum()
        sample_weights = pt_weights[train_dataset.targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    dataloader_class = torch.utils.data.DataLoader

    if not use_sampler:
        train_dataloader = dataloader_class(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False,
        )
    else:
        train_dataloader = dataloader_class(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
        )

    val_dataloader = dataloader_class(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False,
    )

    if use_class_weights:
        class_counts = np.bincount(train_dataset.targets)
        sampler_power = sampler_power
        pt_weights = (1.0 / class_counts) ** sampler_power
        pt_weights /= pt_weights.sum()
        class_weights = torch.from_numpy(pt_weights).float().to("cuda")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    trained_model, history = train_model(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        model_save_path=model_save_path,
        experiment_name=experiment_name,
        run_name=run_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=learning_rate,
        model_name=model_name,
        num_classes=num_classes,
        mixup_alpha=mixup_alpha,
        mixup_p=mixup_p,
        use_sampler=use_sampler,
        use_class_weights=use_class_weights,
        optimizername=optimizer_name,
        sampler_power=sampler_power,
    )

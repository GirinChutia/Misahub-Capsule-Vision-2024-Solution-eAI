"""
Evaluate EfficientViT model on capsule endoscopy images

This script evaluates the performance of a pre-trained EfficientViT model on a test set of capsule endoscopy images.

The script takes as input the path to the test folder, the path to the best model checkpoint, the number of classes in the classification task, and the name of the output Excel file.

The script outputs a Excel file containing the predicted labels for each image in the test set.

"""

import argparse
import glob
import os
import torch
from simple_train import load_model
from eval_model import get_image_paths, single_image_inference, infer_folder, evaluate_model


def parse_arguments():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: a namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate EfficientViT model on capsule endoscopy images")

    parser.add_argument("--test_folder", type=str, required=True, help="Path to test folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the best model checkpoint")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the classification task")
    parser.add_argument("--output_file", type=str, default="submission.xlsx", help="Name of the output Excel file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Get the list of test images
    test_image_paths = glob.glob(os.path.join(args.test_folder, "*.jpg"))

    print(f"Number of test images: {len(test_image_paths)}")

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

    # Perform inference on the test images
    df = infer_folder(model, test_image_paths, type="test", device=device)

    # Save the results to an Excel file
    df.to_excel(args.output_file, index=False)

    print(f"Results saved to {args.output_file}")

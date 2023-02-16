import os
import torch
import argparse
from ultralytics import YOLO


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', None))

    args, _ = parser.parse_known_args()

    model = YOLO('yolov8n.pt')

    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.model.state_dict(), f)

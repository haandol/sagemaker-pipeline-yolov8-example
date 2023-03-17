import os
import yaml
import json
import torch
import shutil
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the Estimator
    parser.add_argument('--model', type=str, default=os.environ.get('SM_HP_MODEL', 'yolov8n'))
    parser.add_argument('--experiment', type=str, default=os.environ['SM_HP_EXPERIMENT'])
    parser.add_argument('--epochs', type=int, default=os.environ['SM_HP_EPOCHS'])
    parser.add_argument('--batch', type=int, default=os.environ['SM_HP_BATCH'])
    parser.add_argument('--imgsz', type=int, default=os.environ['SM_HP_IMGSZ'])
    parser.add_argument('--seed', type=int, default=os.environ['SM_HP_SEED'])

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--valid', type=str, default=os.environ['SM_CHANNEL_VALID'])

    args, _ = parser.parse_known_args()

    # Load the model
    model = YOLO(f'{args.model}.pt')

    # modify data conf
    data_conf_path = '/opt/ml/code/data.yaml' 
    with open(data_conf_path, 'r') as fp:
        data = yaml.load(fp.read(), Loader=yaml.BaseLoader)
        data['train'] = str(Path(args.train) / 'images')
        data['val'] = str(Path(args.valid) / 'images')

    with open(data_conf_path, 'w') as fp:
        yaml.dump(data, fp)
        logger.info(f'Updated data conf: {json.dumps(data, indent=2)}')

    # Training
    device = ','.join(','.join(map(lambda x: str(x), range(1)))) if torch.cuda.is_available() else 'cpu'
    logger.info(f'Training on device: {device}')
    model.train(
        data=data_conf_path,
        name=args.experiment,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        device=device,
        exist_ok=True,
    )

    # Save the model as PyTorch model
    best_pt_path = os.path.join('runs', 'detect', args.experiment, 'weights', 'best.pt')
    output_pt_path = os.path.join(args.model_dir, 'model.pt')
    logger.info(f'Copying model from {best_pt_path} to {output_pt_path}...')

    shutil.copyfile(best_pt_path, output_pt_path)

    logger.info('Training complete!')
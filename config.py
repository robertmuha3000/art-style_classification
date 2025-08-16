import os
import torch

BATCH_SIZE = 32
NUM_WORKERS = max(os.cpu_count() - 1, 0)
PERSISTENT_WORKERS = NUM_WORKERS > 0
HEAD_EPOCHS = 3
TRAIN_EPOCHS = 12
CSV_PATH = "data/classes.csv"
IMAGE_ROOT = "painting_classify/archive-3/"
MODEL_PATH = "models/resnet34_wikiart_class_uncertainty.pth"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
TRAINING_LOSS = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

import os
import torch

BATCH_SIZE = 32
NUM_WORKERS = max(os.cpu_count() - 1, 0)
PERSISTENT_WORKERS = NUM_WORKERS > 0
HEAD_EPOCHS = 3
TRAIN_EPOCHS = 12
CSV_PATH = "classes.csv"
IMAGE_ROOT = "painting_classify/archive-3/"
MODEL_PATH = "resnet34_wikiart_third_final.pth"
LABEL_ENCODER_PATH = "label_encoder.pkl"
TRAINING_LOSS = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
import torch

class hypers():
  seed = 123
  torch.manual_seed(seed)

  # Hyperparameters etc.
  LEARNING_RATE = 2e-5
  DEVICE = "cuda" if torch.cuda.is_available else "cpu"
  # DEVICE = "CUDA"
  BATCH_SIZE = 16 # 64 in original
  WEIGHT_DECAY = 0
  EPOCHS = 10
  NUM_WORKERS = 2
  PIN_MEMORY = True
  LOAD_MODEL = False
  LOAD_MODEL_v8 = False
  LOAD_MODEL_FILE = "./overfit.pth.tar"
  NUM_CLASSES=1
  NUM_BOXES=2
  SPLIT=7
  DATASET_PATH = "LOCAL" #LOCAL/DRIVE
  

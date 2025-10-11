# config.py
import os
import torch

# Find the absolute path of the directory where this file is located
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # --- Security ---
    # IMPORTANT: Change this to a long, random string in production!
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a-super-secret-key-that-you-should-change')

    # --- Database ---
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 
        'sqlite:///' + os.path.join(basedir, 'instance', 'database.db'))
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Model & File Paths ---
    MODEL_PATHS = {
        'ResNet-18': os.path.join(basedir, 'models', 'resnet18_blood_cell_best.pth'),
        'MobileNetV2': os.path.join(basedir, 'models', 'mobilenet_v2_best.pth')
    }
    CLASS_LABELS = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'neutrophil']
    NUM_CLASSES = len(CLASS_LABELS)
    AGREEMENT_CONFIDENCE_THRESHOLD = 85.0 
    TEMP_UPLOAD_DIR = os.path.join('app', 'static', 'uploads', 'temp')
    PROCESSED_DATA_DIR = os.path.join(basedir, 'data_processed')
    IMAGE_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
"""
Default configuration for the Flowers Recognition project.
"""

# Data Configuration
DATA_CONFIG = {
    'raw_data_path': 'data/raw/flowers',
    'processed_data_path': 'data/processed',
    'sample_images_path': 'data/sample_images',
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
    'image_size': (224, 224)
}

# Model Configuration
MODEL_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'num_classes': 5,
    'class_names': ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
}

# Training Configuration
TRAIN_CONFIG = {
    'models_dir': 'models',
    'results_dir': 'results',
    'save_best_only': True,
    'monitor_metric': 'val_accuracy'
}

# Web App Configuration
APP_CONFIG = {
    'host': 'localhost',
    'port': 8080,
    'debug': False,
    'model_path': 'models/best_model.h5'
}

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

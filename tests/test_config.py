"""
Test configuration setup and loading.
"""

import unittest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from configs.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    APP_CONFIG,
    AUGMENTATION_CONFIG
)

class TestConfig(unittest.TestCase):
    """Test cases for configuration settings."""
    
    def test_data_config(self):
        """Test data configuration settings."""
        self.assertIsInstance(DATA_CONFIG, dict)
        self.assertIn('raw_data_path', DATA_CONFIG)
        self.assertIn('processed_data_path', DATA_CONFIG)
        self.assertEqual(sum([
            DATA_CONFIG['train_split'],
            DATA_CONFIG['val_split'],
            DATA_CONFIG['test_split']
        ]), 1.0)

    def test_model_config(self):
        """Test model configuration settings."""
        self.assertIsInstance(MODEL_CONFIG, dict)
        self.assertIn('batch_size', MODEL_CONFIG)
        self.assertIn('epochs', MODEL_CONFIG)
        self.assertEqual(len(MODEL_CONFIG['class_names']), MODEL_CONFIG['num_classes'])

    def test_train_config(self):
        """Test training configuration settings."""
        self.assertIsInstance(TRAIN_CONFIG, dict)
        self.assertIn('models_dir', TRAIN_CONFIG)
        self.assertIn('results_dir', TRAIN_CONFIG)

    def test_app_config(self):
        """Test web app configuration settings."""
        self.assertIsInstance(APP_CONFIG, dict)
        self.assertIn('host', APP_CONFIG)
        self.assertIn('port', APP_CONFIG)
        self.assertIsInstance(APP_CONFIG['port'], int)

    def test_augmentation_config(self):
        """Test data augmentation configuration settings."""
        self.assertIsInstance(AUGMENTATION_CONFIG, dict)
        self.assertIn('rotation_range', AUGMENTATION_CONFIG)
        self.assertIn('horizontal_flip', AUGMENTATION_CONFIG)
        self.assertIsInstance(AUGMENTATION_CONFIG['horizontal_flip'], bool)

if __name__ == '__main__':
    unittest.main()

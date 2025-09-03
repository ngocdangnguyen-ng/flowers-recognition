"""
Model evaluation module for flower recognition.
This module contains functions for evaluating model performance.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, model, test_generator, class_names):
        """
        Initialize the evaluator with a model and test data.
        
        Args:
            model (tf.keras.Model): The trained model to evaluate
            test_generator: The test data generator
            class_names (list): List of class names
        """
        self.model = model
        self.test_generator = test_generator
        self.class_names = class_names
        self.evaluation_results = None
    
    def evaluate(self):
        """
        Perform comprehensive model evaluation.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Reset generator
        self.test_generator.reset()
        
        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(
            self.test_generator, verbose=0
        )
        
        # Generate classification report
        report = classification_report(
            true_classes, 
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        
        # Store results
        self.evaluation_results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predicted_classes,
            'true_labels': true_classes,
            'prediction_probabilities': predictions
        }
        
        return self.evaluation_results
    
    def get_class_performance(self):
        """
        Get per-class performance metrics.
        
        Returns:
            dict: Dictionary containing per-class metrics
        """
        if self.evaluation_results is None:
            self.evaluate()
        
        report = self.evaluation_results['classification_report']
        
        class_metrics = {}
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                class_metrics[class_name] = {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                }
        
        return class_metrics
    
    def analyze_errors(self, num_samples=5):
        """
        Analyze misclassified samples.
        
        Args:
            num_samples (int): Number of misclassified samples to analyze per class
            
        Returns:
            dict: Dictionary containing misclassified examples
        """
        if self.evaluation_results is None:
            self.evaluate()
        
        true_classes = self.evaluation_results['true_labels']
        pred_classes = self.evaluation_results['predictions']
        probs = self.evaluation_results['prediction_probabilities']
        
        # Find misclassified samples
        misclassified = np.where(true_classes != pred_classes)[0]
        
        error_analysis = {
            'total_errors': len(misclassified),
            'error_rate': len(misclassified) / len(true_classes),
            'samples': []
        }
        
        # Analyze samples
        for idx in misclassified[:num_samples]:
            error_analysis['samples'].append({
                'index': int(idx),
                'true_class': self.class_names[true_classes[idx]],
                'predicted_class': self.class_names[pred_classes[idx]],
                'confidence': float(probs[idx][pred_classes[idx]])
            })
        
        return error_analysis
    
    def get_confidence_metrics(self):
        """
        Calculate confidence-based metrics.
        
        Returns:
            dict: Dictionary containing confidence metrics
        """
        if self.evaluation_results is None:
            self.evaluate()
        
        probs = self.evaluation_results['prediction_probabilities']
        max_probs = np.max(probs, axis=1)
        
        return {
            'mean_confidence': float(np.mean(max_probs)),
            'median_confidence': float(np.median(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs))
        }

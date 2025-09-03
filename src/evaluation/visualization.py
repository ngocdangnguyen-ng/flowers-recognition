"""
Visualization module for model evaluation results.
This module contains functions for creating plots and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class EvaluationVisualizer:
    def __init__(self, class_names):
        """
        Initialize the visualizer.
        
        Args:
            class_names (list): List of class names
        """
        self.class_names = class_names
        plt.style.use('default')
        
    def plot_training_history(self, history, title="Model Training History"):
        """
        Plot training history curves.
        
        Args:
            history (dict): Training history dictionary
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history['accuracy'], label='Training')
        ax1.plot(history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history['loss'], label='Training')
        ax2.plot(history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        return plt.gcf()
    
    def plot_class_performance(self, class_metrics, title="Class Performance"):
        """
        Plot per-class performance metrics.
        
        Args:
            class_metrics (dict): Dictionary of class metrics
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        metrics = ['precision', 'recall', 'f1-score']
        class_names = list(class_metrics.keys())
        
        data = []
        for class_name in class_names:
            data.append([
                class_metrics[class_name][metric] 
                for metric in metrics
            ])
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, data[:, i], width, label=metric)
        
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_distribution(self, probabilities, title="Prediction Confidence Distribution"):
        """
        Plot distribution of prediction confidences.
        
        Args:
            probabilities (array): Array of prediction probabilities
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        max_probs = np.max(probabilities, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=50, edgecolor='black')
        plt.title(title)
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for statistics
        plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probs):.3f}')
        plt.axvline(np.median(max_probs), color='green', linestyle='--',
                   label=f'Median: {np.median(max_probs):.3f}')
        
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_error_analysis(self, error_samples, title="Error Analysis"):
        """
        Plot examples of misclassified samples.
        
        Args:
            error_samples (list): List of dictionaries containing error information
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_samples = len(error_samples)
        fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
        
        if n_samples == 1:
            axes = [axes]
        
        for ax, sample in zip(axes, error_samples):
            # Here you would normally load and display the image
            # For now, we'll just show the information
            ax.text(0.5, 0.5, 
                   f"True: {sample['true_class']}\n" + \
                   f"Pred: {sample['predicted_class']}\n" + \
                   f"Conf: {sample['confidence']:.3f}",
                   ha='center', va='center')
            ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig

"""
Metrics for anomaly detection evaluation
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyMetrics:
    """Metrics computation for anomaly detection"""
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores
            threshold: Threshold used (if any)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Additional metrics
        metrics['accuracy'] = (metrics['tp'] + metrics['tn']) / len(y_true)
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0.0
        metrics['false_positive_rate'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0.0
        metrics['false_negative_rate'] = metrics['fn'] / (metrics['fn'] + metrics['tp']) if (metrics['fn'] + metrics['tp']) > 0 else 0.0
        
        if threshold is not None:
            metrics['threshold'] = threshold
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold based on a metric
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            metric: Metric to optimize ('f1', 'precision', 'recall', 'roc_auc')
            
        Returns:
            Tuple of (optimal_threshold, metrics_dict)
        """
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        if metric == 'f1':
            # Find threshold that maximizes F1
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            optimal_idx = np.argmax(precision)
        elif metric == 'recall':
            optimal_idx = np.argmax(recall)
        else:
            # Default to F1
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
        
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else np.median(y_scores)
        
        # Compute predictions with optimal threshold
        y_pred = (y_scores > optimal_threshold).astype(int)
        metrics_dict = self.compute_metrics(y_true, y_pred, y_scores, optimal_threshold)
        
        return optimal_threshold, metrics_dict
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    
    def plot_score_distribution(
        self,
        y_scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None
    ):
        """Plot distribution of anomaly scores"""
        plt.figure(figsize=(10, 6))
        
        if y_true is not None:
            normal_scores = y_scores[y_true == 0]
            anomaly_scores = y_scores[y_true == 1]
            
            plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', color='blue')
            plt.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', color='red')
        else:
            plt.hist(y_scores, bins=50, alpha=0.7, color='blue')
        
        if threshold is not None:
            plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.3f}')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    
    def print_metrics_summary(self, metrics: Dict[str, float]):
        """Print formatted metrics summary"""
        print("\n" + "="*50)
        print("METRICS SUMMARY")
        print("="*50)
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print(f"FPR:          {metrics['false_positive_rate']:.4f}")
        print(f"FNR:          {metrics['false_negative_rate']:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
        print(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")
        if 'threshold' in metrics:
            print(f"\nThreshold: {metrics['threshold']:.4f}")
        print("="*50 + "\n")


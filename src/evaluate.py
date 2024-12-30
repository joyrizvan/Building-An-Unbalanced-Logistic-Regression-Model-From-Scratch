from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class Metrics:
    def evaluate_model(self, y_pred, y_test, print_values: bool = False):
        y_test = y_test.ravel()  # or y_test.flatten()
        # Confusion Matrix
        tp = np.sum((y_pred == 1) & (y_test == 1))  # True Positives
        tn = np.sum((y_pred == 0) & (y_test == 0))  # True Negatives
        fp = np.sum((y_pred == 1) & (y_test == 0))  # False Positives
        fn = np.sum((y_pred == 0) & (y_test == 1))  # False Negatives

        # Confusion matrix
        confusion_matrix = np.array([[tp, fp], [fn, tn]])

        # Accuracy
        accuracy = (tp + tn) / len(y_test)

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Score
        score = (fn * 50) + (fp * 5)

        if print_values:
            print(f"Accuracy - {accuracy*100:.2f}%")
            print(f"Recall - {recall*100:.2f}%")
            print(f"Precision - {precision*100:.2f}%")
            print(
                f"Our total score with 50 for false negative and 5 for false positive - {score}"
            )
            self.confusion_matrix(y_test, y_pred)

        return y_pred, confusion_matrix, accuracy, precision, recall

    def classification_report(self, y_test: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generates a classification report based on true and predicted labels.
        Parameters:
        y_test (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        Returns:
        str: The classification report as a string.
        """
        report = classification_report(y_test, y_pred, output_dict=False)
        print("\nClassification Report:")
        print(report)
        return report

    def confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates and prints the confusion matrix.
        Parameters:
        y_test (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        Returns:
        np.ndarray: The confusion matrix.
        """
        cm = confusion_matrix(y_test, y_pred)
        # print("Confusion Matrix:")
        self.plot_confusion_matrix(cm, "Confusion Matrix")
        return cm

    def plot_confusion_matrix(self, cm, title):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Low Risk (0)", "Predicted High Risk (1)"],
            yticklabels=["Actual Low Risk (0)", "Actual High Risk (1)"],
        )
        plt.title(title)
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.show()

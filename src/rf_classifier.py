from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import numpy as np

class Classifier:
    def __init__(self, no_est: int = 100):
        self.classifier = RandomForestClassifier(n_estimators=no_est)

    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains random forest classifier based on the given training data and labels
        :param X_train: Numpy array containing the features of the training data
        :param y_train: Numpy array containing the labels of the training data
        """

        self.classifier.fit(X_train, y_train)

    def compute_metrics(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Computes PR AUC score of the trained model based on the test data + labels
        :param X_test: Numpy array containing the features of the test data
        :param y_test: Numpy array containing the labels of the test data
      
        :return: PR AUC score
        """

        y_pred = self.classifier.predict_proba(X_test)[:, 1]

        pr_auc_score = average_precision_score(y_test, y_pred)

        return pr_auc_score

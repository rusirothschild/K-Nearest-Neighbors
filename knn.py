import numpy as np

class KNN:
    def __init__(self, k: int = 3, classification: bool = True):
        self.k = k
        self.classification = classification

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def eucl_distance(self, x):  
        distances = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(distances)[:self.k]
        self.top_k = self.y[idx]
        self.zeros = np.sum(self.top_k == 0)
        self.ones = np.sum(self.top_k == 1)
        prob_0 = (self.zeros / self.k) * 100
        prob_1 = (self.ones / self.k) * 100
        self.probability = np.array([prob_0, prob_1]) 
        return self

    def predict_proba(self, X):
        probas = []
        for x in X:
            self.eucl_distance(x)
            probas.append([self.zeros / self.k, self.ones / self.k])
        return np.array(probas)

    def predict(self, X):
        predictions = []
        for x in X:
            self.eucl_distance(x)
            if self.classification:
                pred = 0 if self.probability[0] > self.probability[1] else 1
            else:
                pred = float(np.mean(self.top_k))
            predictions.append(pred)
        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)
        if self.classification:
            return (y_pred == y_true).sum() / len(y_true)
        return 1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()
from sklearn.base import BaseEstimator
class ConstantClassifier(BaseEstimator):
    def __init__(self, constant):
        self.constant = constant
    
    def fit(self, X, y):
        return self

    def predict(self, X):
        return [self.constant] * len(X)

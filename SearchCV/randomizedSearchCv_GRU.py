import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from build_model.build_gru import *

class PytorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, vocab_size, embedding_dim, units, num_layers, drop_out, activation, batch_normalization, bidirectional, lr=0.001, epochs=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.num_layers=num_layers
        self.drop_out=drop_out
        self.activation=activation
        self.batch_normalization=batch_normalization
        self.bidirectional=bidirectional
        self.lr=lr
        self.epochs=epochs
        self.device=torch.device('cude' if torch.cuda.is_available() else 'cpu')
        self.model=GRU(vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            gru_units=self.units,
            num_layers=self.num_layers,
            dropout=self.drop_out,
            activation=self.activation,
            batch_normalization=self.batch_normalization,
            bidirectional=self.bidirectional).to(self.device)
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr)
    def fit(self, X, y):
        tensorDataset=TensorDataset(X, y)
        trainDataLoader=DataLoader(tensorDataset, batch_size=16, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for inputs, labels in trainDataLoader:
                output=self.model(inputs.to(self.device))
                self.optimizer.zero_grad()
                loss=self.criterion(output, labels.to(self.device))
                loss.backward()
                self.optimizer.step()
        self.classes_=np.unique(y)
        return self
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs=self.model(X.to(self.device))
            _, predicted=torch.max(outputs, 1)
        return predicted.numpy()
    def score(self, X, y):
        predicted=self.predict(X)
        return accuracy_score(predicted, y.to(self.device))

if __name__=="__main__":
    pass










'''

class PyTorchModelWrapper(BaseEstimator):
    def __init__(self, vocab_size, embedding_dim, num_layers, activation, batch_normalization, bidirectional, lr=0.001, epochs=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.bidirectional = bidirectional
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GRU(self.vocab_size, self.embedding_dim, self.num_layers, 
                         self.activation, self.batch_normalization, self.bidirectional).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(self, X, y):
        trainer = trainer()  # Giả sử bạn đã có hàm train
        trainer.train(self.model, X, y, self.criterion, self.optimizer, self.epochs, self.device)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X.to(self.device))
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

from sklearn.model_selection import RandomizedSearchCV

def perform_random_search_pytorch(model_name, model, param_dist, X_train, y_train, n_iter=3, cv=None):
    """
    Perform Randomized Search for hyperparameter tuning on a PyTorch model.
    """

    # Set up RandomizedSearchCV
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter, cv=cv, verbose=0)

    # Fit the RandomizedSearchCV to find the best parameters
    randomized_search.fit(X_train, y_train)

    # Display results
    print(f"\n{model_name} Model Results After Hyperparameter Tuning:\n")
    
    # Get all the results from the randomized search
    cv_results = randomized_search.cv_results_
    for params, mean_test_score, rank in zip(cv_results["params"], cv_results["mean_test_score"], cv_results["rank_test_score"]):
        print(f"Params: {params} - Mean Test Score: {mean_test_score:.2f} - Rank: {rank}")

    # Get the best parameters
    best_parameters = randomized_search.best_params_
    print(f"\nBest Parameters for {model_name} model: {best_parameters}\n")

    # Retrieve the best model after search
    best_model = randomized_search.best_estimator_

    return best_parameters, best_model
param_dist = {
    'embedding_dim': randint(64, 256),
    'num_layers': randint(1, 4),
    'activation': [None, nn.ReLU(), nn.Tanh()],
    'batch_normalization': [True, False],
    'bidirectional': [True, False],
    'lr': uniform(0.0001, 0.01),
    'epochs': randint(5, 20)
}

# Khởi tạo wrapper cho mô hình PyTorch
model_wrapper = PyTorchModelWrapper(vocab_size=5000, embedding_dim=128, num_layers=2, activation=None, 
                                    batch_normalization=True, bidirectional=False)

# Gọi hàm tìm kiếm
best_params, best_model = perform_random_search_pytorch('GRU', model_wrapper, param_dist, X_train, y_train, n_iter=10, cv=3)
'''
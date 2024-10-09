from sklearn.base import BaseEstimator
import torch
import torch.optim as optim
import numpy as np

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

'''
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

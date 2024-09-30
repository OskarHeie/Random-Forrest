import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    """
    A class to represent a random forrest. Creates n number of trees and predicts
    based on majority vote from the trees.
    Inputs: n_estimators: number of trees in the forrest
            max_depth: max depth for each tree
            criterion: either gini or entropy
            max_features: None, log2 or sqrt, the number of features to use for
            each tree, which features is randomized for each tree
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.tree_lis = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This functions learns n decision trees given (continuous) features X and (integer) labels y.
        """

        self.tree_lis = []
        for i in range(self.n_estimators):
            indexes = np.random.choice(range(X.shape[0]), X.shape[0], replace=True)
            new_X = X[indexes, :]
            new_y = y[indexes]

            tree = DecisionTree(self.max_depth, self.criterion, self.max_features)
            tree.fit(new_X, new_y)
            self.tree_lis.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        Decided by majority vote, votes collected from each tree.
        """

        pred_lis = []
        for tree in self.tree_lis:
            pred_lis.append(tree.predict(X))
        pred_mat = np.array(pred_lis)

        pred = []
        for i in range(pred_mat[0, :].size):
            values, counts = np.unique(pred_mat[:, i], return_counts=True)
            ind = np.argmax(counts)
            pred.append(values[ind])

        return np.array(pred)



if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="entropy", max_features='sqrt'
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")

import math
import numpy as np
from typing import Self

def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    values, count = np.unique(y, return_counts=True)
    return count


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index (a number between 0 and 0.5) of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    
    y = y.flatten()
    counts = count(y)
    prob = counts / len(y)
    gini = 1 - np.sum((prob)**2)
    
    return gini


def entropy(y: np.ndarray) -> float:
    """
    Return the entropy (a number between 0 and 1.0) of a given NumPy array y.
    """
    y = y.flatten()
    counts = count(y)
    prob = counts / len(y)
    entropy = -np.sum((prob * np.log2(prob)))

    return entropy

def split_boolean(x: np.ndarray, value: float, over_under: str) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """

    if over_under == 'under':
        return x <= value
    elif over_under == 'over':
        return x > value
    else:
        raise ValueError(f"Invalid direction: {over_under}")

def split(X: np.ndarray, split_value: float, index: int) -> np.ndarray:
    """
    Takes an array and splits it into two arrays based on input column index
    and input split value.
    """
    left_split = X[X[:, index] <= split_value]
    
    right_split = X[X[:, index] > split_value]
    
    return left_split, right_split

def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    values, count = np.unique(y, return_counts=True)
    index = np.argmax(count)
    return values[index]

def information_gain(X: np.ndarray, y: np.ndarray, split_index: int, criterion: str) -> int:
    """
    Takes a 2D array X with the features and 1D array y with labels, split_index
    which is the index of column to look at information gain from and criterion
    which is either gini or entropy and returns the a positive int information
    gain for this split when splitting on the mean of the column.
    """

    split_mean = np.mean(X[:, split_index])

    left_X = split_boolean(X[:, split_index], split_mean, 'under')
    right_X = split_boolean(X[:, split_index], split_mean, 'over')

    if criterion == 'gini':
        pre_split_measure = gini_index(y)
        left_measure = gini_index(y[left_X])
        right_measure = gini_index(y[right_X])

    else:
        pre_split_measure = entropy(y)
        left_measure = entropy(y[left_X])
        right_measure = entropy(y[right_X])

    n = len(y)
    n_left = sum(left_X)
    n_right = sum(right_X)

    if n_left == 0 or n_right == 0:
        return 0

    weighted_measure = (n_left / n) * left_measure + (n_right / n) * right_measure

    info_gain = pre_split_measure - weighted_measure
    #I get really small negative values which are likely errors so i set them to zero instead
    #if they are small enough
    tolerance = 1e-12 
    if abs(info_gain) < tolerance:
        info_gain = 0

    return info_gain

def ID3(X: np.ndarray, y: np.ndarray, criterion: str, max_depth: int, depth: int, feature_indexes: list):
    """
    The ID3 algorithm. Takes X: a 2D array with all features, y: a 1D array with all labels, 
    criterion which is either gini or entropy, max_depth: the depth to stop at depth: the 
    current depth and feature_indexes: a list with indexes of features to look at in X. If 
    all labels are the same return a node with a prediction. If all features have the same
    value return a node predicting the most common label for y. Else find the feature which
    gains the most information to split on and run ID3 on the split dataset.
    """

    values = np.unique(y)
    if len(values) == 1:
        node = Node(value=values.item())
        return node
    
    if len(count(X[:, feature_indexes])) == 1 or depth == max_depth:
        node = Node(value=most_common(y))
        return node

    else:
        index = -1
        max_information_gain = 0
        for i in feature_indexes:
            info_gain = information_gain(X, y, i, criterion)
            if info_gain > max_information_gain:
                index = i
                max_information_gain = info_gain

        if index == -1:
            node = Node(value=most_common(y))
            return node

        mean = X[:, index].mean()
        left_X, right_X = split(X, mean, index)
        left_y = y[split_boolean(X[:, index], mean, 'under')]
        right_y = y[split_boolean(X[:, index], mean, 'over')]
        
        node = Node(index, mean, ID3(left_X, left_y, criterion, max_depth, depth+1, feature_indexes), ID3(right_X, right_y, criterion, max_depth, depth+1, feature_indexes))
        return node

class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        # Return True iff the node is a leaf node
        return self.value is not None


class DecisionTree:
    """
    A class to represent a Decision Tree.
    Inputs: max_depth: level (int) for which the tree is to stop building at.
            criterion: either gini or entropy
            max_features: None, log2 or sqrt, the number of features to use
    """

    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
        max_features: None | str = None,
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """
        if self.max_features == 'sqrt':
            indexes = np.random.choice(range(X.shape[1]), math.floor(math.sqrt(X.shape[1])), replace=False)
        
        elif self.max_features == 'log2':
            indexes = np.random.choice(range(X.shape[1]), math.floor(math.log2(X.shape[1])), replace=False)
            
        else:
            indexes = [i for i in range(X.shape[1])]
        
        self.root = ID3(X, y, self.criterion, self.max_depth, 0, indexes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        pred = []
        for row in X:
            node = self.root
            while not node.is_leaf():
                feature = row[node.feature]
                if feature <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            pred.append(node.value)

        return np.array(pred)

    
    def print_tree(self, node, level=0):
        """
        A method to create a readable version of the tree in the consol
        """
        if node.value == None:
            print("  " * level + str(node.threshold))
            self.print_tree(node.left, (level + 1))
            self.print_tree(node.right, (level + 1))
        else:
            print("  " * level + str(node.value))

    


if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    #print(gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])))
    #print(entropy(np.array([1, 1, 2, 2, 3, 3, 4, 4,4])))
    #print(split(np.array([1, 2, 3, 4, 5, 2]), 3))
    #print(most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])))
    #print(ID3(np.array([[1,1,1,0], [1,1,1,1]]), np.array([1,1,1,1]), 'entropy', None).value)
    #print(ID3(np.array([[1,1,1,1], [1,1,1,1]]), np.array([1,0,0,0]), 'entropy', None).value)
    #node = ID3(np.array([[1,1,1,1], [0,0,0,0]]), np.array([1,0]), 'entropy', None)
    #print(node.value)
    #print(node.left.value)
    #print(node.right.value)
    #print(information_gain(np.array([[1,1,1], [1,5,2], [0,3,8]]), np.array([1,1,0]), 0))

    seed = 1

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=None, criterion="entropy", max_features='log2')
    rf.fit(X_train, y_train)
    #rf.print_tree(rf.root)
    

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}") 

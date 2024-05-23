import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from itertools import compress
import openml
from sklearn import compose, impute, preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances

def proximityMatrix(model, X, normalize=True):
    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1 * np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat


def save_model(clf, path):
    with open(path, 'wb') as f:
        joblib.dump(clf, f)


def load_model(path):
    with open(path, 'rb') as f:
        clf = joblib.load(f)
        return clf


def save_proximity(matrix, name):
    np.save(f'./proximities/{name}.npy', matrix)

def load_proximity( name):
    return np.load(f'./proximities/{name}.npy')


def load_breast_cancer_data(random_seed=172119):
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data  # Features
    y = data.target  # Target labels (0: malignant, 1: benign)
    feature_names = data.feature_names  # Feature names

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # split into training & testing data
    indices = np.arange(n_samples)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X,
                                                                             y,
                                                                             indices,
                                                                             test_size=0.2,
                                                                             random_state=random_seed, stratify=y)

    # Data as tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    return n_features, n_classes, X, y, X_train, X_test, y_train, y_test, idx_train, idx_test


def load_openml_data(dataset_id, data_type='ordinal', random_seed=172119):
    openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=True,
                                                 download_features_meta_data=False)
    X, y, categorical_indicator, attribute_names = openml_dataset.get_data(dataset_format="dataframe",
                                                                           target=openml_dataset.default_target_attribute)
    cat_attribs = list(compress(attribute_names, categorical_indicator))
    cont_attribs = list(compress(attribute_names, [not elem for elem in categorical_indicator]))
    eeg, *_ = openml_dataset.get_data()
    n_samples = eeg.shape[0]

    if data_type == 'ordinal':
        cat_encoder = preprocessing.OrdinalEncoder()
    else:
        cat_encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")

    cat_transformer = Pipeline(
        steps=[
            ("cat_encoder", cat_encoder),
            ("cat_impute", impute.SimpleImputer(strategy="most_frequent"))
        ]
    )
    ct = compose.ColumnTransformer(
        [
            (
                "label_encoder",
                preprocessing.OrdinalEncoder(),
                [openml_dataset.default_target_attribute],
            ),
            ('cat_transformer', cat_transformer, cat_attribs),
            (
                "num_imputer",
                impute.SimpleImputer(strategy="median"),
                cont_attribs,
            ),
        ]
    )

    eeg = ct.fit_transform(eeg)

    eeg_train, eeg_test = train_test_split(eeg, test_size=0.2, random_state=random_seed, stratify=y)


    y_train = eeg_train[:, 0]
    n_classes = len(np.unique(y))
    X_train = np.delete(eeg_train, 0, 1)
    n_features = X_train.shape[1]
    y_test = eeg_test[:, 0]
    X_test = np.delete(eeg_test, 0, 1)
    # Data as tensors

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    X = torch.cat([X_train, X_test])
    y = torch.cat([y_train, y_test])
    idx_train = torch.LongTensor(np.array(range(0, X_train.shape[0])))
    idx_test = torch.LongTensor(np.array(range(X_train.shape[0], X.shape[0])))
    return n_features, n_classes, X, y, X_train, X_test, y_train, y_test, idx_train, idx_test








def jaccard_similarity_matrix(X):
    n = X.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = np.sum(np.minimum(X[i], X[j]))
            union = np.sum(np.maximum(X[i], X[j]))
            similarity_matrix[i, j] = intersection / union
    return similarity_matrix


def cosine_similarity_matrix(X):
    # Compute dot product of X with its transpose
    dot_product = np.dot(X, X.T)

    # Compute magnitude of each row
    row_magnitude = np.sqrt(np.diag(dot_product))

    # Compute cosine similarity matrix
    cosine_matrix = dot_product / np.outer(row_magnitude, row_magnitude)

    return cosine_matrix


def rbf_kernel_matrix(X, gamma=0.01):
    # Compute pairwise squared Euclidean distances
    dists = euclidean_distances(X, X)
    # Compute RBF kernel matrix
    rbf_matrix = np.exp(-gamma * dists)

    return rbf_matrix


# def RF_GAP(X, best_rf_model, grid_search, X_train, X_test, y):
#     Prox_org = proximityMatrix(best_rf_model, X, normalize=True)
#     rf_gap = RFGAP(min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
#                    min_samples_split=grid_search.best_params_['min_samples_split'],
#                    n_estimators=grid_search.best_params_['n_estimators'],
#                    prediction_type='classification',
#                    matrix_type='dense',
#                    force_symmetric=True)
#     rf_gap.fit(X_train, y_train)
#     rf_gap_proximities = rf_gap.get_proximities()
#     min_prox = np.min(rf_gap_proximities)
#     max_prox = np.max(rf_gap_proximities)
#     Prox_best = (rf_gap_proximities - min_prox) / (max_prox - min_prox)
#     n_train = X_train.shape[0]
#     n_test = X_test.shape[0]
#     Prox_best[:n_test, :n_test] = Prox_org[:n_test, :n_test]
#     return Prox_best

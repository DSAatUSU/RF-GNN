import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from utils import *
from sklearn.ensemble import RandomForestClassifier


openml.config.apikey = 'a48e8796d88855d777a16536371daac2'

classifier_type = 'rf'

BASE_SAVE_DIR = f'./models/{classifier_type}'
if not os.path.exists(BASE_SAVE_DIR):
    os.makedirs(BASE_SAVE_DIR)

if not os.path.exists(f'./proximities'):
    os.makedirs(f'./proximities')

if not os.path.exists(f'./results'):
    os.makedirs(f'./results')

dataset_list = [43757, 31, 446, 720, 825, 853, 902, 915, 941, 955, 983, 1006, 1012, 1167, 1498, 40705,
                40710, 40981, 43255, 43942, 44098, 23, 475, 1557, 40663, 1478, 1053, 32, 4534, 6, 1486, 1461, 1590,
                182, 40701, 300]

random_seeds = [172119, 42, 12, 7889, 1015]
results = []

rf_param_grid = {
    'n_estimators': [50, 100, 200, 500, 700, 1000],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 20, 50, 80, 100, 150, 200, 300, 500]
}

for index, item in enumerate(dataset_list):
    dataset = f'{item}'

    scores = []

    ## Get the data ready

    n_features, n_classes, X, y, X_train, X_test, y_train, y_test, idx_train, idx_test = load_openml_data(
        int(dataset))

    print(
        f'Applying {classifier_type} on dataset {dataset} with #instances: {X.shape[0]} and #features: {X.shape[1]} and #classes: {n_classes}')

    for random_seed in random_seeds:

        classifier = RandomForestClassifier(random_state=random_seed)
        param_grid = rf_param_grid

        # Create a GridSearchCV object
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='f1_weighted',
                                   n_jobs=-1,
                                   verbose=0)

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Print the best parameters found by the grid search
        print("Best Parameters:", grid_search.best_params_)

        # Get the best model from the grid search
        best_model = grid_search.best_estimator_

        save_model(best_model, path=f'{BASE_SAVE_DIR}/{classifier_type}_{dataset}_{random_seed}.joblib')
        best_model = load_model(
            path=f'{BASE_SAVE_DIR}/{classifier_type}_{dataset}_{random_seed}.joblib')
        y_pred = best_model.predict(X_test)
        # Evaluate the best model on the test set
        if n_classes > 2:
            test_f1 = f1_score(y_pred, y_test, average='weighted')
        else:
            test_f1 = f1_score(y_pred, y_test)

        print(f"Test F1-score with Best {classifier_type} Model: {test_f1:.4f}")

        Prox_org = proximityMatrix(best_model, X, normalize=True)
        save_proximity(Prox_org, f'{classifier_type}_prox_{dataset}_{random_seed}')

        scores.append(test_f1)
    mean = sum(scores) / len(scores)
    variance = sum([((x - mean) ** 2) for x in scores]) / len(scores)
    res = variance ** 0.5
    results.append([dataset, round(mean, 4), round(res, 4)])

    df = pd.DataFrame(results, columns=['dataset', f'{classifier_type}_mean', f'{classifier_type}_std'])
    df.to_csv(f'./results/{classifier_type}_results.csv', index=False)

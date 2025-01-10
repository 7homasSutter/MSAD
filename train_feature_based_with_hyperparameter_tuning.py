########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : train_feature_based
#
########################################################################


import argparse
import os
from time import perf_counter
import re
from collections import Counter
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from utils.timeseries_dataset import create_splits, TimeseriesDataset
from eval_feature_based import eval_feature_based
from utils.evaluator import save_classifier
from utils.config import *

names = {
		"knn": "Nearest Neighbors",
		"svc_linear": "Linear SVM",
		"decision_tree": "Decision Tree",
		"random_forest": "Random Forest",
		"mlp": "Neural Net",
		"ada_boost": "AdaBoost",
		"bayes": "Naive Bayes",
		"qda": "QDA",
}

classifiers = {
		"knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
		"svc_linear": LinearSVC(C=0.025, verbose=True),
		"decision_tree": DecisionTreeClassifier(max_depth=5),
		"random_forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1, verbose=True),
		"mlp": MLPClassifier(alpha=1, max_iter=1000, verbose=True),
		"ada_boost": AdaBoostClassifier(),
		"bayes": GaussianNB(),
		"qda": QuadraticDiscriminantAnalysis(),
}

# Define parameter grids for each classifier
param_grids = {
    "knn": {
        "n_neighbors": [2, 4, 5, 8, 12, 16, 32, 128, 256, 512],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
        "p": [1, 2]
    },
    "svc": {"C": [0.01, 0.025]},
    "svc_linear": {"C": [0.01, 0.025]},
        "decision_tree": {
        "max_depth": [3, 5, 7, 9, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        "min_samples_split": [2, 5, 10, 20, 50, 100],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "max_features": ["auto", "sqrt", "log2", None],
        "criterion": ["gini", "entropy"]
    },
    "random_forest": {"n_estimators": [50, 100, 500], "max_depth": [3, 5, 7, 9]},
    "mlp": {"alpha": [0.0001, 0.001, 0.01, 0.1], "max_iter": [500, 1000]},
    "ada_boost": {"n_estimators": [50, 100, 200]},
    "bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    "qda": {"reg_param": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
}


def train_feature_based(data_path, classifier_name, split_per=0.7, seed=None, read_from_file=None, eval_model=False, path_save=None, sample_fraction=0.3):
    # Set up
    window_size = int(re.search(r'\d+', data_path).group())
    training_stats = {}

    if "nt" in os.name:
        original_dataset = data_path.split('\\')[:-1]
        original_dataset = '\\'.join(original_dataset)
    else:
        original_dataset = data_path.split('/')[:-1]
        original_dataset = '/'.join(original_dataset)

    # Load the splits
    train_set, val_set, test_set = create_splits(
        original_dataset,
        split_per=split_per,
        seed=seed,
        read_from_file=read_from_file,
    )
    train_indexes = [x[:-4] for x in train_set]
    val_indexes = [x[:-4] for x in val_set]
    test_indexes = [x[:-4] for x in test_set]

    # Read tabular data
    data = pd.read_csv(data_path, index_col=0)

    # Reindex them
    data_index = list(data.index)
    new_index = [tuple(x.rsplit('.', 1)) for x in data_index]
    new_index = pd.MultiIndex.from_tuples(new_index, names=["name", "n_window"])
    data.index = new_index

    # Normalize delimiters in data.index
    data.index = data.index.set_levels([level.str.replace('/', '\\') for level in data.index.levels])

    # Normalize delimiters in val_indexes and test_indexes
    train_indexes = [x.replace('/', '\\') for x in train_indexes]
    val_indexes = [x.replace('/', '\\') for x in val_indexes]
    test_indexes = [x.replace('/', '\\') for x in test_indexes]
    print(f"Training on {len(train_indexes)} instances, validating on {len(val_indexes)} instances, testing on {len(test_indexes)} instances")

    # Create subsets
    training_data = data.loc[data.index.get_level_values("name").isin(train_indexes)]
    val_data = data.loc[data.index.get_level_values("name").isin(val_indexes)]
    test_data = data.loc[data.index.get_level_values("name").isin(test_indexes)]
    print(f"Training on {len(training_data)} instances, validating on {len(val_data)} instances, testing on {len(test_data)} instances")

    # Split data from labels
    y_train, X_train = training_data['label'], training_data.drop('label', 1)
    y_val, X_val = val_data['label'], val_data.drop('label', 1)
    y_test, X_test = test_data['label'], test_data.drop('label', 1)

    if X_train.shape[0] == 0:
        raise ValueError("Training set is empty. Please check the data and splitting parameters.")

    # Sample a subset of the training data
    sample_size = int(len(y_train) * sample_fraction)
    print(f"Sampling {sample_size} instances from the training set using {sample_fraction} sample_fraction...")
    sample_indices = np.random.choice(len(y_train), sample_size, replace=False)
    X_train_sample = X_train.iloc[sample_indices]
    y_train_sample = y_train.iloc[sample_indices]

    # Select the classifier
    classifier = classifiers[classifier_name]
    clf_name = classifier_name

    # Hyperparameter tuning
    param_grid = param_grids[classifier_name]
    grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1, verbose=3)
    print(f'----------------------------------\nTuning hyperparameters for {names[classifier_name]}...')
    tic = perf_counter()
    grid_search.fit(X_train_sample, y_train_sample)
    toc = perf_counter()
    best_classifier = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Hyperparameter tuning time: {toc - tic:.3f} secs")

    # Save tuning results to CSV
    tuning_results = pd.DataFrame(grid_search.cv_results_)
    tuning_results['tuning_time'] = toc - tic
    tuning_results.to_csv(os.path.join(save_done_training, f"{classifier_name}_tuning_results.csv"), index=False)

    # Fit the classifier with the best parameters
    print(f'Training {names[classifier_name]} with best parameters...')
    tic = perf_counter()
    best_classifier.fit(X_train, y_train)
    toc = perf_counter()

    # Print training time
    training_stats["training_time"] = toc - tic
    print(f"training time: {training_stats['training_time']:.3f} secs")

    # Print valid accuracy and inference time
    tic = perf_counter()
    classifier_score = best_classifier.score(X_val, y_val)
    toc = perf_counter()
    training_stats["val_acc"] = classifier_score
    training_stats["avg_inf_time"] = ((toc-tic)/X_val.shape[0]) * 1000
    print(f"valid accuracy: {training_stats['val_acc']:.3%}")
    print(f"inference time: {training_stats['avg_inf_time']:.3} ms")

    # Save training stats
    classifier_name = f"{clf_name}_tuned_{window_size}"
    if read_from_file is not None and "unsupervised" in read_from_file:
        classifier_name += f"_{os.path.basename(read_from_file).replace('unsupervised_', '')[:-len('.csv')]}"
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    df = pd.DataFrame.from_dict(training_stats, columns=["training_stats"], orient="index")
    df.to_csv(os.path.join(save_done_training, f"{classifier_name}_{timestamp}.csv"))

    # Save pipeline
    saving_dir = os.path.join(path_save, classifier_name) if classifier_name.lower() not in path_save.lower() else path_save
    saved_model_path = save_classifier(best_classifier, saving_dir, fname=None)

    # Evaluate on test set or val set
    if eval_model:
        eval_set = test_indexes if len(test_indexes) > 0 else val_indexes
        print(f"Evaluating {classifier_name} on eval set: {len(eval_set)}, Data path: {data_path}, Model name {classifier_name}, Model path: {saved_model_path}, Path save: {path_save_results}, Fnames: {len(eval_set)}")
        eval_feature_based(
            data_path=data_path,
            model_name=classifier_name,
            model_path=saved_model_path,
            path_save=path_save_results,
            fnames=eval_set,
        )

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='train_feature_based',
		description='Script for training the traditional classifiers',
	)
	parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
	parser.add_argument('-c', '--classifier', type=str, help='classifier to run', required=True)
	parser.add_argument('-sp', '--split_per', type=float, help='split percentage for train and val sets', default=0.7)
	parser.add_argument('-s', '--seed', type=int, help='seed for splitting train, val sets (use small number)', default=None)
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="results/weights")

	args = parser.parse_args()

	# Option to all classifiers
	if args.classifier == 'all':
		clf_list = list(classifiers.keys())
	else:
		clf_list = [args.classifier]

	if not os.path.exists(save_done_training):
		os.makedirs(save_done_training)


	for classifier in clf_list:
		train_feature_based(
			data_path=args.path,
			classifier_name=classifier,
			split_per=args.split_per, 
			seed=args.seed,
			read_from_file=args.file,
			eval_model=args.eval_true,
			path_save=args.path_save
		)

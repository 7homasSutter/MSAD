@echo off

REM Nearest Neighbors
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=knn --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true

REM Linear SVM
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=svc_linear --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true

REM Decision Tree
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=decision_tree --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true

REM Random Forest
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=random_forest --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true

REM Neural Net
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=mlp --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true

REM AdaBoost
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=ada_boost --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true

REM Naive Bayes
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=bayes --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true

REM QDA
python train_feature_based.py --path=.\data\TSB_16\TSFRESH_TSB_16.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_16.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_32\TSFRESH_TSB_32.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_32.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_64\TSFRESH_TSB_64.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_64.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_128\TSFRESH_TSB_128.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_128.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_256\TSFRESH_TSB_256.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_256.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_512\TSFRESH_TSB_512.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_512.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_768\TSFRESH_TSB_768.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_768.csv --path_save=.\results\weights\ --eval-true
python train_feature_based.py --path=.\data\TSB_1024\TSFRESH_TSB_1024.csv --classifier=qda --split_per=0.7 --file=.\experiments\supervised_splits\split_TSB_1024.csv --path_save=.\results\weights\ --eval-true


REM Nearest Neighbors
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=knn --model_path=results\weights\knn_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=knn --model_path=results\weights\knn_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=knn --model_path=results\weights\knn_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=knn --model_path=results\weights\knn_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=knn --model_path=results\weights\knn_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=knn --model_path=results\weights\knn_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=knn --model_path=results\weights\knn_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=knn --model_path=results\weights\knn_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

REM Linear SVM
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=svc_linear --model_path=results\weights\svc_linear_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=svc_linear --model_path=results\weights\svc_linear_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=svc_linear --model_path=results\weights\svc_linear_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=svc_linear --model_path=results\weights\svc_linear_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=svc_linear --model_path=results\weights\svc_linear_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=svc_linear --model_path=results\weights\svc_linear_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=svc_linear --model_path=results\weights\svc_linear_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=svc_linear --model_path=results\weights\svc_linear_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

REM Decision Tree
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=decision_tree --model_path=results\weights\decision_tree_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=decision_tree --model_path=results\weights\decision_tree_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=decision_tree --model_path=results\weights\decision_tree_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=decision_tree --model_path=results\weights\decision_tree_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=decision_tree --model_path=results\weights\decision_tree_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=decision_tree --model_path=results\weights\decision_tree_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=decision_tree --model_path=results\weights\decision_tree_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=decision_tree --model_path=results\weights\decision_tree_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

REM Random Forest
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=random_forest --model_path=results\weights\random_forest_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=random_forest --model_path=results\weights\random_forest_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=random_forest --model_path=results\weights\random_forest_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=random_forest --model_path=results\weights\random_forest_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=random_forest --model_path=results\weights\random_forest_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=random_forest --model_path=results\weights\random_forest_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=random_forest --model_path=results\weights\random_forest_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=random_forest --model_path=results\weights\random_forest_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

REM Neural Net
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=mlp --model_path=results\weights\mlp_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=mlp --model_path=results\weights\mlp_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=mlp --model_path=results\weights\mlp_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=mlp --model_path=results\weights\mlp_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=mlp --model_path=results\weights\mlp_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=mlp --model_path=results\weights\mlp_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=mlp --model_path=results\weights\mlp_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=mlp --model_path=results\weights\mlp_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

REM AdaBoost
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=ada_boost --model_path=results\weights\ada_boost_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=ada_boost --model_path=results\weights\ada_boost_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=ada_boost --model_path=results\weights\ada_boost_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=ada_boost --model_path=results\weights\ada_boost_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=ada_boost --model_path=results\weights\ada_boost_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=ada_boost --model_path=results\weights\ada_boost_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=ada_boost --model_path=results\weights\ada_boost_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=ada_boost --model_path=results\weights\ada_boost_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

REM Naive Bayes
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=bayes --model_path=results\weights\bayes_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=bayes --model_path=results\weights\bayes_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=bayes --model_path=results\weights\bayes_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=bayes --model_path=results\weights\bayes_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=bayes --model_path=results\weights\bayes_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=bayes --model_path=results\weights\bayes_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=bayes --model_path=results\weights\bayes_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=bayes --model_path=results\weights\bayes_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

REM QDA
python eval_feature_based.py --data=data\TSB_16\TSFRESH_TSB_16.csv --model=qda --model_path=results\weights\qda_16\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_16.csv
python eval_feature_based.py --data=data\TSB_32\TSFRESH_TSB_32.csv --model=qda --model_path=results\weights\qda_32\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_32.csv
python eval_feature_based.py --data=data\TSB_64\TSFRESH_TSB_64.csv --model=qda --model_path=results\weights\qda_64\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_64.csv
python eval_feature_based.py --data=data\TSB_128\TSFRESH_TSB_128.csv --model=qda --model_path=results\weights\qda_128\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_128.csv
python eval_feature_based.py --data=data\TSB_256\TSFRESH_TSB_256.csv --model=qda --model_path=results\weights\qda_256\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_256.csv
python eval_feature_based.py --data=data\TSB_512\TSFRESH_TSB_512.csv --model=qda --model_path=results\weights\qda_512\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_512.csv
python eval_feature_based.py --data=data\TSB_768\TSFRESH_TSB_768.csv --model=qda --model_path=results\weights\qda_768\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_768.csv
python eval_feature_based.py --data=data\TSB_1024\TSFRESH_TSB_1024.csv --model=qda --model_path=results\weights\qda_1024\ --path_save=results\raw_predictions\ --file=experiments\supervised_splits\split_TSB_1024.csv

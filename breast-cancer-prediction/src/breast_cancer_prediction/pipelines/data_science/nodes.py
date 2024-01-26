from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import logging
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def split_data(data: pd.DataFrame, parameters: dict):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - parameters (dict): A dictionary containing parameters for data splitting.

    Returns:
    - tuple: A tuple containing X_train, y_train, X_test, y_test, X_train_val, X_val, y_train_val, y_val.
    """
    if parameters["use_all_features"]:
        X = data.drop([parameters["target_feature"]], axis=1)
        y = data[parameters["target_feature"]]
    else:
        X = data[parameters["features"]]
        y = data[parameters["target_feature"]]

    y = (y == "M").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], stratify=y, random_state=parameters["random_state"]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=parameters["test_size"], stratify=y_train, random_state=parameters["random_state"]
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_transformer() -> ColumnTransformer:
    numeric_features = list(range(30))
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    return preprocessor

def get_best_model(grid_search):
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_params, best_estimator, best_score

def knn_model(X_train, y_train, X_val, y_val, preprocessor: ColumnTransformer) -> GridSearchCV:
    knn = KNeighborsClassifier()

    param_grid_knn = {
        'knn__n_neighbors': np.arange(1, 31, 2),
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1,2]
    }

    pipe_knn = Pipeline(steps=[('preprocessor', preprocessor),
                                ('knn', knn)])

    gs_knn = GridSearchCV(estimator=pipe_knn,
                          param_grid=param_grid_knn,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1,
                          verbose=1)

    gs_knn.fit(X_train, np.ravel(y_train))

    best_params, best_estimator, best_score = get_best_model(gs_knn)


    logger.info(f"Best knn hyperparameters: {best_params}")
    logger.info(f"Best knn accuracy: {best_score}")
    logger.info(f"Best knn estimator: {best_estimator}")
    
    y_pred_train = gs_knn.predict(X_train)
    y_pred_val = gs_knn.predict(X_val)
    
    # Log parameters
    mlflow.log_params({
        **gs_knn.best_params_
    })

    # Log metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'accuracy': accuracy_score(y_val, y_pred_val),
        'auc': roc_auc_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'f1': f1_score(y_val, y_pred_val)
    }
    mlflow.log_metrics(metrics)
    metrics.update({"model": "knn_model", "parameters": gs_knn.best_params_})

    # Log model artifact
    mlflow.sklearn.log_model(best_estimator, "knn_model")

    return gs_knn, {key: (str(value) if key == 'parameters' else value) for key, value in metrics.items()}


def svm_model(X_train, y_train, X_val, y_val, preprocessor: ColumnTransformer) -> GridSearchCV:
    svm = SVC(random_state=1, max_iter=500, probability=True)

    param_grid_svm = {'svm__C': [10**i for i in range(-3, 4)],
                    'svm__gamma': [10**i for i in range(-3, 4)],
                    'svm__kernel': ['linear', 'rbf']
    }

    pipe_svm = Pipeline(steps=[('preprocessor', preprocessor),
                                ('svm', svm)])

    gs_svm = GridSearchCV(estimator=pipe_svm,
                          param_grid=param_grid_svm,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1,
                          verbose=1)

    gs_svm.fit(X_train, np.ravel(y_train))

    best_params, best_estimator, best_score = get_best_model(gs_svm)

    logger.info(f"Best SVM hyperparameters: {best_params}")
    logger.info(f"Best SVM accuracy: {best_score}")

    y_pred_train = gs_svm.predict(X_train)
    y_pred_val = gs_svm.predict(X_val)

    # Log parameters
    mlflow.log_params({
        **gs_svm.best_params_
    })

    # Log metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'accuracy': accuracy_score(y_val, y_pred_val),
        'auc': roc_auc_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'f1': f1_score(y_val, y_pred_val)
    }
    mlflow.log_metrics(metrics)
    metrics.update({"model": "svm_model", "parameters": gs_svm.best_params_})

    # Log model artifact
    mlflow.sklearn.log_model(best_estimator, "svm_model")

    return gs_svm, {key: (str(value) if key == 'parameters' else value) for key, value in metrics.items()}

def random_forest_model(X_train, y_train, X_val, y_val, preprocessor: ColumnTransformer) -> GridSearchCV:
    rf = RandomForestClassifier()

    param_grid_rf = {
        "rf__n_estimators": list(range(100, 1001, 50)), 
        "rf__criterion": ["gini", "entropy"], 
        "rf__max_depth": [None, 2, 4, 6, 8, 10], 
        "rf__min_samples_split": [2, 4, 6, 8, 10], 
        "rf__min_samples_leaf": [1, 2, 4, 6, 8, 10],
        'rf__max_features': ['auto', 'sqrt', 'log2'], 
        'rf__bootstrap': [True, False]  
    }

    pipe_rf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('rf', rf)])

    rs_rf = RandomizedSearchCV(estimator=pipe_rf, 
                           param_distributions=param_grid_rf, 
                           cv=10, 
                           n_iter=70, 
                           scoring="accuracy", 
                           n_jobs=-1, 
                           verbose=1,
                           random_state=1)

    rs_rf.fit(X_train, np.ravel(y_train))

    best_params, best_estimator, best_score = get_best_model(rs_rf)

    logger.info(f"Best Random Forest hyperparameters: {best_params}")
    logger.info(f"Best Random Forest accuracy: {best_score}")

    y_pred_train = rs_rf.predict(X_train)
    y_pred_val = rs_rf.predict(X_val)

    # Log parameters
    mlflow.log_params({
        **rs_rf.best_params_
    })

    # Log metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'accuracy': accuracy_score(y_val, y_pred_val),
        'auc': roc_auc_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'f1': f1_score(y_val, y_pred_val)
    }
    mlflow.log_metrics(metrics)
    metrics.update({"model": "rf_model", "parameters": rs_rf.best_params_})

    # Log model artifact
    mlflow.sklearn.log_model(best_estimator, "rf_model")

    return rs_rf, {key: (str(value) if key == 'parameters' else value) for key, value in metrics.items()}


def find_best_model(X_train, y_train, X_val, y_val, preprocessor, parameters: dict):
    models = [
        knn_model(X_train, y_train, X_val, y_val, preprocessor),
        svm_model(X_train, y_train, X_val, y_val, preprocessor),
        random_forest_model(X_train, y_train, X_val, y_val, preprocessor),
    ]

    metrics_models = []
    best_f1_score = 0
    best_model = None

    for model, metrics in models:
        metrics_models.append(metrics)

        if metrics['f1'] > best_f1_score:
            best_f1_score = metrics[parameters["metric"]]
            best_model = model

    logger.info(f"List metric models: {metrics_models}")
    return best_model, metrics_models

# def find_best_model(X_train, y_train, X_val, y_val, X_test, y_test):
#     numeric_features = list(range(30))
#     preprocessor = _preprocess_transformer(numeric_features)

#     models = [
#         knn_model(X_train, y_train, X_val, y_val, preprocessor),
#         svm_model(X_train, y_train, X_val, y_val, preprocessor)
#     ]

#     # Get best models
#     best_models = []
#     for model in models:
#         best_params, best_estimator, best_score = get_best_model(model)
#         best_models.append((best_params, best_estimator, best_score))

#     logger.info(best_models)
#      # Select best model
#     best_model = max(best_models, key=lambda x: x[2])
#     best_estimator = best_model[1]

#     # logger.info(best_model)
#     # logger.info(best_estimator)

#     # Retrain best model on entire training data
#     best_estimator.fit(X_train, y_train)

#     # Evaluate best model on test data
#     test_score = best_estimator.score(X_test, y_test)
#     logger.info(f"Test accuracy: {test_score}")

#     return {"best_model": best_model, "best_estimator": best_estimator}






"""
def random_forest_model(X_train_val, y_train_val, preprocessor) -> GridSearchCV[Pipeline]:
    rf = RandomForestClassifier()

    param_grid_rf = {'n_estimators': [50, 100, 200],
                     'max_depth': [None, 10, 20, 30]
    }

    pipe_rf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('rf', rf)])

    gs_rf = GridSearchCV(estimator=pipe_rf,
                         param_grid=param_grid_rf,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1,
                         verbose=1)

    gs_rf.fit(X_train_val, y_train_val)

    return gs_rf

def xgboost_model(X_train_val, y_train_val, preprocessor) -> GridSearchCV[Pipeline]:
    xgb = XGBClassifier()

    param_grid_xgb = {'n_estimators': [50, 100, 200],
                      'max_depth': [3, 5, 7],
                      'learning_rate': [0.01, 0.1, 0.2]
    }

    pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                ('xgb', xgb)])

    gs_xgb = GridSearchCV(estimator=pipe_xgb,
                          param_grid=param_grid_xgb,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1,
                          verbose=1)

    gs_xgb.fit(X_train_val, y_train_val)

    return gs_xgb

def lightgbm_model(X_train_val, y_train_val, preprocessor) -> GridSearchCV[Pipeline]:
    lgbm = LGBMClassifier()

    param_grid_lgbm = {'n_estimators': [50, 100, 200],
                       'max_depth': [3, 5, 7],
                       'learning_rate': [0.01, 0.1, 0.2]
    }

    pipe_lgbm = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('lgbm', lgbm)])

    gs_lgbm = GridSearchCV(estimator=pipe_lgbm,
                           param_grid=param_grid_lgbm,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1,
                           verbose=1)

    gs_lgbm.fit(X_train_val, y_train_val)

    return gs_lgbm

"""
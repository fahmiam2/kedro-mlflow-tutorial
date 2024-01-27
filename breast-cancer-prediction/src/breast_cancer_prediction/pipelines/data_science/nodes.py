from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
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
import matplotlib.pyplot as plt
import seaborn as sns

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

# validation
def predict_validation_model(model, X_train, X_val, y_train, y_val, model_name):
    best_params, best_estimator, best_score = get_best_model(model)
    
    logger.info(f"Best {model_name} hyperparameters: {best_params}")
    logger.info(f"Best {model_name} accuracy: {best_score}")
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_val_prob = model.predict_proba(X_val)[:, 1] 
    
    # Log parameters
    # mlflow.log_params({
    #     **best_params
    # })

    # Log metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'accuracy': accuracy_score(y_val, y_pred_val),
        'auc': roc_auc_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'f1': f1_score(y_val, y_pred_val)
    }
    # mlflow.log_metrics(metrics)
    metrics.update({"model": f"{model_name}_model", "parameters": best_params})

    # Log model artifact
    # mlflow.sklearn.log_model(best_estimator, f"{model_name}_model")

    return pd.Series(y_pred_val), pd.Series(y_pred_val_prob), {key: (str(value) if key == 'parameters' else value) for key, value in metrics.items()}

# test
def predict_test_model(model, X_train, X_test, y_train, y_test):

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_test_prob = model.predict_proba(X_test)[:, 1] 
    
    # Log parameters
    # mlflow.log_params({
    #     **best_params
    # })

    # Log metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'accuracy': accuracy_score(y_test, y_pred_test),
        'auc': roc_auc_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test)
    }
    # mlflow.log_metrics(metrics)
    # metrics.update({"model": f"{model_name}_model", "parameters": best_params})

    # Log model artifact
    # mlflow.sklearn.log_model(best_estimator, f"{model_name}_model")

    return metrics, pd.Series(y_pred_test), pd.Series(y_pred_test_prob)

# models
def train_knn_model(X_train, y_train, preprocessor: ColumnTransformer) -> GridSearchCV:
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
    
    gs_knn.fit(X_train, y_train)

    return gs_knn, "knn"

def train_svm_model(X_train, y_train, preprocessor: ColumnTransformer) -> GridSearchCV:
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
    
    gs_svm.fit(X_train, y_train)

    return gs_svm, "svm"

def train_random_forest_model(X_train, y_train, preprocessor: ColumnTransformer) -> RandomizedSearchCV:
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
    
    rs_rf.fit(X_train, y_train)

    return rs_rf, "rf"

def find_best_model(X_train, y_train, X_val, y_val, parameters: dict, *models):

    y_pred_val_df = pd.DataFrame()
    y_pred_val_prob_df = pd.DataFrame()
    metrics_models = []
    best_metric_score = 0
    best_model = None

    for model, model_name in zip(models, parameters["model_names"]):
        y_pred_val, y_pred_val_prob, metrics = predict_validation_model(model, X_train, y_train, X_val, y_val, model_name)
        metrics_models.append(metrics)

        if metrics[parameters["metric"]] > best_metric_score:
            best_metric_score = metrics[parameters["metric"]]
            best_model = model

        y_pred_val_df[f'{model_name}'] = y_pred_val
        y_pred_val_prob_df[f'{model_name}'] = y_pred_val_prob

    logger.info(f"List metric models: {metrics_models}")
    return best_model, metrics_models, y_pred_val_df, y_pred_val_prob_df




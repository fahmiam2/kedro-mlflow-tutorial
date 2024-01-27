from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data, preprocess_transformer, train_knn_model, train_svm_model, 
    train_random_forest_model, find_best_model, predict_test_model
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_diagnostic", "params:model_options"],
                outputs=["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
                name="split_data_node"
            ),
            node(
                func=preprocess_transformer,
                inputs=None,
                outputs="preprocessor",
                name="preprocess_transformer"
            ),
            node(
                func=train_knn_model,
                inputs=["X_train", "y_train", "preprocessor"],
                outputs=["knn_object_model", "knn"]
            ),
            node(
                func=train_svm_model,
                inputs=["X_train", "y_train", "preprocessor"],
                outputs=["svm_object_model", "svm"]
            ),
            node(
                func=train_random_forest_model,
                inputs=["X_train", "y_train", "preprocessor"],
                outputs=["rf_object_model", "rf"]
            )
        ]
    )

def create_find_best_model(**kwargs):
    return pipeline(
        [
            node(
                func=find_best_model,
                inputs=["X_train", "X_val", "y_train", "y_val", "params:model_evaluation", "knn_object_model", "svm_object_model", "rf_object_model"],
                outputs=["best_model", "metrics_models", "y_pred_val_df", "y_pred_val_prob_df"]
            )
        ]
    )

def create_evaluate_model_test(**kwargs):
    return pipeline(
        [
            node(
                func=predict_test_model,
                inputs=["best_model", "X_train", "X_test", "y_train", "y_test"],
                outputs=["metrics", "y_pred_test", "y_pred_test_prob"]
            )
        ]
    )
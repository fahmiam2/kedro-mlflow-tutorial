from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, preprocess_transformer, svm_model, knn_model, random_forest_model, find_best_model

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
            # node(
            #     func=knn_model,
            #     inputs=["X_train", "y_train", "X_val", "y_val", "preprocessor"],
            #     outputs=["knn_model", "knn_model_metric"],
            #     name="train_knn_model",
            # ),
            # node(
            #     func=svm_model,
            #     inputs=["X_train", "y_train", "X_val", "y_val", "preprocessor"],
            #     outputs=["svm_model", "svm_model_metric"],
            #     name="train_svm_model",
            # ),
            # node(
            #     func=random_forest_model,
            #     inputs=["X_train", "y_train", "X_val", "y_val", "preprocessor"],
            #     outputs=["random_forest_model", "random_forest_model_metric"],
            #     name="train_random_forest_model",
            # ),
            node(
                func=find_best_model,
                inputs=["X_train", "y_train", "X_val", "y_val", "preprocessor", "params:model_evaluation"],
                outputs=["best_model", "log_compare_model"],
                name="find_best_model"
            )
        ]
    )
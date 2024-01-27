from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    plot_confusion_matrix,
    plot_roc_curve
)

models = ["knn", "svm", "rf"]

def create_pipeline_val(**kwargs) -> Pipeline:
    node_list = []

    for model in models:

        node_list.append(
            node(
                func=plot_confusion_matrix,
                inputs=["y_val", "y_pred_val_df", model],
                outputs=f"{model}_model_cm"
            )
        )

        node_list.append(
            node(
                func=plot_roc_curve,
                inputs=["y_val", "y_pred_val_prob_df", model],
                outputs=f"{model}_model_roc_auc"
            )
        )
    return Pipeline(nodes=(node_list))

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=plot_confusion_matrix,
                inputs=["y_test", "y_pred_test"],
                outputs="best_model_cm"
            ),
            node(
                func=plot_roc_curve,
                inputs=["y_test", "y_pred_test_prob"],
                outputs="best_model_roc_auc"
            )
        ]
    )
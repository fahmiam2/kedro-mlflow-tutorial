# src/breast_cancer_prediction/pipelines/data_preprocessing/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import drop_unnecessary_columns

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=drop_unnecessary_columns,
                inputs=["diagnostic", "params:drop_columns"],
                outputs="preprocessed_diagnostic",
                name='drop_unnecessary_columns'
            )
        ]
    )
"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from breast_cancer_prediction.pipelines import data_preprocessing as dp
from breast_cancer_prediction.pipelines import data_science as ds

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_preprocessing = dp.create_pipeline()
    data_science = ds.create_pipeline()

    return {
        "dp": data_preprocessing,
        "ds": data_science,
        "__default__": (
            data_preprocessing
            + data_science
        ),
    }

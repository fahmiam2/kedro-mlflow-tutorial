"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from breast_cancer_prediction.pipelines import data_preprocessing as dp
from breast_cancer_prediction.pipelines import data_science as ds
from breast_cancer_prediction.pipelines import reporting as rp

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_preprocessing = dp.create_pipeline()
    data_science = ds.create_pipeline()
    find_best_model = ds.create_find_best_model()
    evaluate_model_test = ds.create_evaluate_model_test()
    reporting = rp.create_pipeline()
    reporting_val = rp.create_pipeline_val()

    return {
        "dp": data_preprocessing,
        "ds": data_science,
        "fm": find_best_model,
        "ts": evaluate_model_test,
        "rp": reporting,
        "rv": reporting_val,
        "__default__": (
            data_preprocessing
            + data_science
            + find_best_model
            + evaluate_model_test
            + reporting
            + reporting_val
        ),
    }

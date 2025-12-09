import os
from src.pipeline import run_training_pipeline
from src.config import MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH

def test_training_pipeline_runs():
    """
    Ensures that running the pipeline creates:
    - trained model file
    - preprocessor file
    - threshold json
    """

    # Run the pipeline
    run_training_pipeline()

    # Assertions
    assert os.path.exists(MODEL_PATH), "Model file was not created."
    assert os.path.exists(PREPROCESSOR_PATH), "Preprocessor file was not created."
    assert os.path.exists(THRESHOLD_PATH), "Threshold file was not created."

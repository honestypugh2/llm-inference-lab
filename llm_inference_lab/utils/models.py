import os
from pydantic import BaseModel

BASE_DIR = os.path.expanduser("~/llm-inference-lab/llm_inference_lab")
MODELS_DIR = f"{BASE_DIR}/models"


class PretrainedModels(BaseModel):
    path: str
    repo: str

pegasus_model = PretrainedModels(
    path=f"{MODELS_DIR}/pegasus-cnn_dailymail", repo="google/pegasus-cnn_dailymail"
)
distilbart_model = PretrainedModels(
    path=f"{MODELS_DIR}/distilbart-cnn-12-6",
    repo="sshleifer/distilbart-cnn-12-6",
)

all_models = [
    pegasus_model,
    distilbart_model,
]

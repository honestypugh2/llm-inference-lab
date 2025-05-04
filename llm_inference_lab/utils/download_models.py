import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from .models import (
    all_models,
)

logger = logging.getLogger(__name__)

def download_models(
    models=False,
    all=False,
):

    """Download and Prepare Models"""
    if models or all:
        download_all_models()

    if not any([models, all]):
        logger.warning("No flags were given, not assets were downloaded")


def download_all_models():
    """Downloads all ML models"""
    for model in all_models:
        download_hugging_face_model(model.repo, model.path)


def download_hugging_face_model(huggingface_repo: str, target_dir: str):
    """Downloads a hugging face model

    Args:
        huggingface_repo (str): hugging face model repo
        target_dir (str): Location of model download
    """
    model_path = Path(target_dir)
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    logger.info("Checking %s", target_dir)
    pytorch_models = list(model_path.glob("**/pytorch_model.bin"))
    if len(pytorch_models) == 0:
        logger.info("Getting model for: %s", huggingface_repo)
        snapshot_download(
            huggingface_repo,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["rust_model.ot", "tf_model.h5"],
        )
    else:
        logger.info("Skipping, Model has been downloaded")



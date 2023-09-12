import os
from pathlib import Path
from typing import Union

import structlog as logging
from llama_cpp import Llama

from src.config import ModelDownloadConfiguration
from src.model.model_downloader import ModelDownloader

logger = logging.get_logger(__name__)


class ModelRetriever:
    def __init__(self, model_downloader: ModelDownloader):
        self.model_downloader = model_downloader

    """Retrieves model from local storage or downloads it if it does not exist."""

    def get_model(self, model_config: ModelDownloadConfiguration, llama_model=Llama):
        """Get model from local storage or download it if it does not exist."""
        path = Path(f"{os.getcwd()}/src/model/downloaded_models/{model_config.folder_name}/")
        if not self._does_model_exist(path, model_config.quantization_model):
            logger.info("Model does not exist for environment. Attempting to download")
            self.model_downloader.download_model(
                model_config.download_link, model_config.folder_name
            )

        logger.info(f"Loading model: {model_config.model_name} from path: {path}")
        model_file: str = path.as_posix() + f"/{model_config.quantization_model}"
        logger.info(f"Instantiating model from: {model_file}")
        model = llama_model(model_path=model_file, **model_config.model_kwargs)
        logger.info(f"Model {model_config.model_name} loaded")
        return model

    def _does_model_exist(
        self, path: Path, quantization_model: Union[str, None]
    ) -> bool:
        """Check if model exists in local storage."""
        if quantization_model is None:
            raise ValueError("Quantization model must be specified")
        model_file = f"{path.as_posix()}/{quantization_model}"
        model_path = Path(model_file)

        return model_path.exists()

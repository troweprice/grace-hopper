import os

import structlog as logging

from src.model.huggingface_downloader import HuggingFaceDownloader

logger = logging.get_logger(__name__)


class ModelDownloader:
    def __init__(
        self, hugging_face_downloader: HuggingFaceDownloader, output_path: str
    ):
        self.hugging_face_downloader = hugging_face_downloader
        self.output_path = output_path

    def download_model(self, download_link: str, folder_name: str):
        """Downloads a LlamaCpp based model to `self.output_path/folder_name` for the provided `download_link`. An example `download_link` is https://huggingface.co/vicuna/ggml-vicuna-13b-1.1/resolve/main/ggml-vic13b-q4_0.bin."""
        try:
            logger.info(f"Downloading model from URL {download_link}")

            filename = download_link.split("/")[-1]
            output_folder = f"{self.output_path}/{folder_name}"
            os.makedirs(output_folder, exist_ok=True)
            self.hugging_face_downloader.get_single_file(
                url=download_link, output_folder=output_folder, filename=filename
            )
            logger.info(f"Successfully downloaded model from URL {download_link}")
        except Exception as exc:
            logger.error(
                f"Failed to download model from URL {download_link}. Error: {exc}"
            )
            raise exc

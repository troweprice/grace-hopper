from src.document.document_sourcer import DocumentSourcer
from src.model.huggingface_downloader import HuggingFaceDownloader

from fastapi import Depends

from src.model.model_downloader import ModelDownloader
from src.model.model_retriever import ModelRetriever
from src.prompt.prompt_builder import PromptBuilder

MODEL_OUTPUT_PATH = "src/model/downloaded_models"


def get_huggingface_downloader():
    """Returns a HuggingFaceDownloader instance."""
    return HuggingFaceDownloader()


def get_model_downloader(
        hugging_face_downloader: HuggingFaceDownloader = Depends(
            get_huggingface_downloader
        ),
):
    """Returns a ModelDownloader instance."""
    return ModelDownloader(
        hugging_face_downloader=hugging_face_downloader,
        output_path=MODEL_OUTPUT_PATH,
    )


def get_model_retriever(
        model_downloader: ModelDownloader = Depends(get_model_downloader),
):
    """Returns a ModelRetriever instance."""
    return ModelRetriever(model_downloader=model_downloader)


def get_document_sourcer():
    """Returns a DocumentSourcer instance."""
    return DocumentSourcer()


def get_prompt_builder():
    """Returns a PromptBuilder instance."""
    return PromptBuilder()


def get_retriever_on_app_start():
    """Returns a ModelRetriever instance."""
    hugging_face_downloader = HuggingFaceDownloader()
    model_downloader = ModelDownloader(
        hugging_face_downloader=hugging_face_downloader,
        output_path=MODEL_OUTPUT_PATH,
    )
    return ModelRetriever(model_downloader)

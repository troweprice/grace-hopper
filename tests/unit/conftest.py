import pytest

from src.config import ModelDownloadConfiguration


@pytest.fixture
def mock_huggingface_model_downloader(mocker):
    return mocker.Mock()


@pytest.fixture
def mock_model_downloader(mocker):
    return mocker.Mock()


@pytest.fixture
def mock_env_config():
    return ModelDownloadConfiguration(
        model_name="TheBloke/Wizard-Vicuna-7B-Uncensored-GGML",
        folder_name="TheBloke_Wizard-Vicuna-7B-Uncensored-GGML",
        download_link="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML/resolve/main/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin",
        quantization_model="Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin",
        model_kwargs={"use_mlock": True, "n_ctx": 1300},
    )

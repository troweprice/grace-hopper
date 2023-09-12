import pytest
from llama_cpp import Llama

from src.model.model_retriever import ModelRetriever


@pytest.fixture
def mock_llama(mocker):
    return mocker.Mock()


def test_get_cpu_model_downloads_model_if_does_not_exist_then_instantiates_model(
    mock_model_downloader, mock_env_config, mock_llama, mocker
):
    # Arrange
    mock_cwd = "/current/working/directory"
    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("os.getcwd", return_value=mock_cwd)
    mock_llama.return_value = Llama

    mock_model_downloader.download_model.return_value = None
    cpu_model_retriever = ModelRetriever(model_downloader=mock_model_downloader)

    # Act
    result = cpu_model_retriever.get_model(mock_env_config, mock_llama)

    # Assert
    mock_model_downloader.download_model.assert_called_once_with(
        mock_env_config.download_link, mock_env_config.folder_name
    )
    expected_model_path = f"{mock_cwd}/src/models/{mock_env_config.folder_name}/{mock_env_config.quantization_model}"
    mock_llama.assert_called_once_with(
        model_path=expected_model_path, **mock_env_config.model_kwargs
    )

    assert result == Llama


def test_get_cpu_model_does_not_downloads_model_if_exists_then_instantiates_model(
    mock_model_downloader, mock_env_config, mock_llama, mocker
):
    # Arrange
    mock_cwd = "/current/working/directory"
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("os.getcwd", return_value=mock_cwd)
    mock_llama.return_value = Llama

    mock_model_downloader.download_model.return_value = None
    cpu_model_retriever = ModelRetriever(model_downloader=mock_model_downloader)

    # Act
    result = cpu_model_retriever.get_model(mock_env_config, mock_llama)

    # Assert
    mock_model_downloader.download_model.assert_not_called()
    expected_model_path = f"{mock_cwd}/src/models/{mock_env_config.folder_name}/{mock_env_config.quantization_model}"
    mock_llama.assert_called_once_with(
        model_path=expected_model_path, **mock_env_config.model_kwargs
    )

    assert result == Llama


def test_get_cpu_model_raises_error_if_quantization_model_is_none(
    mock_model_downloader, mock_env_config, mock_llama, mocker
):
    # Arrange
    mock_env_config.quantization_model = None
    cpu_model_retriever = ModelRetriever(model_downloader=mock_model_downloader)

    # Act
    with pytest.raises(ValueError) as e:
        cpu_model_retriever.get_model(mock_env_config, mock_llama)

    # Assert
    assert str(e.value) == "Quantization model must be specified"

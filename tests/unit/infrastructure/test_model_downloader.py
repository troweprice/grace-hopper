import os

import pytest

from src.model.model_downloader import ModelDownloader


def test_download_cpu_model_creates_the_output_folder_and_call_huggingface_downloader_with_correct_parameters(
    mock_huggingface_model_downloader, mocker
):
    # Arrange
    huggingface_model_link = "https://huggingface.co/vicuna/ggml-vicuna-13b-1.1/resolve/main/ggml-vic13b-q4_0.bin"
    mock_output_path = "/output/path"
    mock_folder_name = "folder_name"

    mocker.patch("os.makedirs")
    mock_huggingface_model_downloader.get_single_file.return_value = None

    model_downloader = ModelDownloader(
        hugging_face_downloader=mock_huggingface_model_downloader,
        output_path=mock_output_path,
    )

    # Act
    model_downloader.download_model(huggingface_model_link, mock_folder_name)

    # Assert
    os.makedirs.assert_called_once_with(
        mock_output_path + "/" + mock_folder_name, exist_ok=True
    )
    mock_huggingface_model_downloader.get_single_file.assert_called_once_with(
        url=huggingface_model_link,
        output_folder="/output/path/folder_name",
        filename="ggml-vic13b-q4_0.bin",
    )


def test_download_cpu_model_raises_an_exception_if_the_downloader_fails(
    mock_huggingface_model_downloader, mocker
):
    # Arrange
    huggingface_model_link = "https://huggingface.co/vicuna/ggml-vicuna-13b-1.1/resolve/main/ggml-vic13b-q4_0.bin"
    mock_output_path = "/output/path"
    mock_folder_name = "folder_name"

    mocker.patch("os.makedirs")
    mock_huggingface_model_downloader.get_single_file.side_effect = Exception

    model_downloader = ModelDownloader(
        hugging_face_downloader=mock_huggingface_model_downloader,
        output_path=mock_output_path,
    )

    # Act & Assert
    with pytest.raises(Exception):
        model_downloader.download_model(huggingface_model_link, mock_folder_name)


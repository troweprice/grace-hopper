from dataclasses import dataclass
from typing import Dict, Optional

from transformers import PreTrainedTokenizer


@dataclass
class ModelDownloadConfiguration:
    """Environment Configuration.

    model_name: str -> The name of the model
    folder_name: str -> The folder where the model will be saved under src/models
    download_link: str -> The HuggingFace link used to download the model file
    model_kwargs: Dict -> Additional parameters that are provided to the model
    quantization_model: str -> The name of the quantization model. This will become the filename of the model when installed
    tokenizer: PreTrainedTokenizer -> Tokenizer necessary for GPU models
    """

    model_name: str
    folder_name: str
    download_link: str
    model_kwargs: Dict
    max_tokens: Optional[int] = 1028
    temperature: Optional[float] = 0.1
    quantization_model: Optional[str] = None
    tokenizer: Optional[PreTrainedTokenizer] = None


def get_config_for_model_download():
    """Get config for environment."""
    return ModelDownloadConfiguration(
        model_name="TheBloke/Wizard-Vicuna-7B-Uncensored-GGML",
        folder_name="TheBloke_Wizard-Vicuna-7B-Uncensored-GGML",
        download_link="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML/resolve/main/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin",
        quantization_model="Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin",
        model_kwargs={"use_mlock": True, "n_ctx": 1300},
    )

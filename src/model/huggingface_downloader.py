"""Downloads models from Hugging Face to models/username_modelname.

Example:
python download-model.py facebook/opt-1.3b

Copied from https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py
"""


import base64
import datetime
import hashlib
import json
import re
from pathlib import Path

import requests
import tqdm
from requests.adapters import HTTPAdapter
from tqdm.contrib.concurrent import thread_map


class HuggingFaceDownloader:
    """Downloads models from Hugging Face to src/models/. Copied from https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py."""

    def __init__(self, max_retries=5):
        self.s = requests.Session()
        if max_retries:
            self.s.mount(
                "https://cdn-lfs.huggingface.co", HTTPAdapter(max_retries=max_retries)
            )
            self.s.mount("https://huggingface.co", HTTPAdapter(max_retries=max_retries))

    def sanitize_model_and_branch_names(self, model, branch):
        """Sanitizes model and branch names."""
        if model[-1] == "/":
            model = model[:-1]

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed."
                )

        return model, branch

    def get_download_links_from_huggingface(self, model, branch, text_only=False):
        """Gets download links from Hugging Face."""
        base = "https://huggingface.co"
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
        classifications = []
        has_pytorch = False
        has_pt = False
        # has_ggml = False
        has_safetensors = False
        is_lora = False
        while True:
            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            r = self.s.get(url, timeout=20)
            r.raise_for_status()
            content = r.content

            dict = json.loads(content)
            if len(dict) == 0:
                break

            for i in range(len(dict)):
                fname = dict[i]["path"]
                if not is_lora and fname.endswith(
                    ("adapter_config.json", "adapter_model.bin")
                ):
                    is_lora = True

                is_pytorch = re.match(r"(pytorch|adapter)_model.*\.bin", fname)
                is_safetensors = re.match(r".*\.safetensors", fname)
                is_pt = re.match(r".*\.pt", fname)
                is_ggml = re.match(r".*ggml.*\.bin", fname)
                is_tokenizer = re.match(r"(tokenizer|ice).*\.model", fname)
                is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any(
                    (is_pytorch, is_safetensors, is_pt, is_ggml, is_tokenizer, is_text)
                ):
                    if "lfs" in dict[i]:
                        sha256.append([fname, dict[i]["lfs"]["oid"]])

                    if is_text:
                        links.append(
                            f"https://huggingface.co/{model}/resolve/{branch}/{fname}"
                        )
                        classifications.append("text")
                        continue

                    if not text_only:
                        links.append(
                            f"https://huggingface.co/{model}/resolve/{branch}/{fname}"
                        )
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append("safetensors")
                        elif is_pytorch:
                            has_pytorch = True
                            classifications.append("pytorch")
                        elif is_pt:
                            has_pt = True
                            classifications.append("pt")
                        elif is_ggml:
                            # has_ggml = True
                            classifications.append("ggml")

            cursor = (
                base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode())
                + b":50"
            )
            cursor = base64.b64encode(cursor)
            cursor = cursor.replace(b"=", b"%3D")

        # If both pytorch and safetensors are available, download safetensors only
        if (has_pytorch or has_pt) and has_safetensors:
            for i in range(len(classifications) - 1, -1, -1):
                if classifications[i] in ["pytorch", "pt"]:
                    links.pop(i)

        return links, sha256, is_lora

    def get_output_folder(self, model, branch, is_lora, base_folder=None):
        """Gets the output folder."""
        if base_folder is None:
            base_folder = "models" if not is_lora else "loras"

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != "main":
            output_folder += f"_{branch}"

        output_folder = Path(base_folder) / output_folder
        return output_folder

    def get_single_file(
        self, url, output_folder, start_from_scratch=False, filename=None
    ):
        """Gets a single file from Hugging Face."""
        filename = filename if filename else Path(url.rsplit("/", 1)[1])
        output_path = Path(output_folder + "/" + filename)
        headers = {}
        mode = "wb"
        if output_path.exists() and not start_from_scratch:
            # Check if the file has already been downloaded completely
            r = self.s.get(url, stream=True, timeout=20)
            total_size = int(r.headers.get("content-length", 0))
            if output_path.stat().st_size >= total_size:
                return

            # Otherwise, resume the download from where it left off
            headers = {"Range": f"bytes={output_path.stat().st_size}-"}
            mode = "ab"

        with self.s.get(url, stream=True, headers=headers, timeout=20) as r:
            r.raise_for_status()  # Do not continue the download if the request was unsuccessful
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1MB
            with open(output_path, mode) as f:
                with tqdm.tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    bar_format="{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6}",
                ) as t:
                    for data in r.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)

    def start_download_threads(
        self, file_list, output_folder, start_from_scratch=False, threads=1
    ):
        """Starts the download threads."""
        thread_map(
            lambda url: self.get_single_file(
                url, output_folder, start_from_scratch=start_from_scratch
            ),
            file_list,
            max_workers=threads,
            disable=True,
        )

    def download_model_files(
        self,
        model,
        branch,
        links,
        sha256,
        output_folder,
        progress_bar=None,
        start_from_scratch=False,
        threads=1,
    ):
        """Downloads the model files."""
        self.progress_bar = progress_bar

        # Creating the folder and writing the metadata
        output_folder.mkdir(parents=True, exist_ok=True)
        metadata = (
            f"url: https://huggingface.co/{model}\n"
            f"branch: {branch}\n"
            f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        )

        sha256_str = "\n".join([f"    {item[1]} {item[0]}" for item in sha256])
        if sha256_str:
            metadata += f"sha256sum:\n{sha256_str}"

        metadata += "\n"
        (output_folder / "huggingface-metadata.txt").write_text(metadata)

        # Downloading the files
        print(f"Downloading the model to {output_folder}")
        self.start_download_threads(
            links, output_folder, start_from_scratch=start_from_scratch, threads=threads
        )

    def check_model_files(self, model, branch, links, sha256, output_folder):
        """Checks if the model files are present and valid."""
        # Validate the checksums
        validated = True
        for i in range(len(sha256)):
            fpath = output_folder / sha256[i][0]

            if not fpath.exists():
                print(f"The following file is missing: {fpath}")
                validated = False
                continue

            with open(output_folder / sha256[i][0], "rb") as f:
                bytes = f.read()
                file_hash = hashlib.sha256(bytes).hexdigest()
                if file_hash != sha256[i][1]:
                    print(f"Checksum failed: {sha256[i][0]}  {sha256[i][1]}")
                    validated = False
                else:
                    print(f"Checksum validated: {sha256[i][0]}  {sha256[i][1]}")

        if validated:
            print("[+] Validated checksums of all model files!")
        else:
            print(
                "[-] Invalid checksums. Rerun download-model.py with the --clean flag."
            )

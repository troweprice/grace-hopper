import argparse
import sys

from src.model.huggingface_downloader import HuggingFaceDownloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL", type=str, default=None, nargs="?")
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Name of the Git branch to download from.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of files to download simultaneously.",
    )
    parser.add_argument(
        "--text-only", action="store_true", help="Only download text files (txt/json)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The folder where the model should be saved.",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Does not resume the previous download."
    )
    parser.add_argument(
        "--check", action="store_true", help="Validates the checksums of model files."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries count when get error in download time.",
    )
    args = parser.parse_args()

    branch = args.branch
    model = args.MODEL

    if model is None:
        print(
            "Error: Please specify the model you'd like to download (e.g. 'python download-model.py facebook/opt-1.3b')."
        )
        sys.exit()

    downloader = HuggingFaceDownloader(max_retries=args.max_retries)
    # Cleaning up the model/branch names
    try:
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)
    except ValueError as err_branch:
        print(f"Error: {err_branch}")
        sys.exit()

    # Getting the download links from Hugging Face
    links, sha256, is_lora = downloader.get_download_links_from_huggingface(
        model, branch, text_only=args.text_only
    )

    # Getting the output folder
    output_folder = downloader.get_output_folder(
        model, branch, is_lora, base_folder=args.output
    )

    if args.check:
        # Check previously downloaded files
        downloader.check_model_files(model, branch, links, sha256, output_folder)
    else:
        # Download files
        downloader.download_model_files(
            model, branch, links, sha256, output_folder, threads=args.threads
        )

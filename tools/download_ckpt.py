import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)
DEFAULT_SYNLAYERS_REPO = "SynLayers/Bbox-caption-8b"


def parse_args():
    parser = argparse.ArgumentParser(description="Download SynLayers pretrained checkpoints.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="SynLayers project root directory.",
    )
    parser.add_argument(
        "--flux-dir",
        default=None,
        help="Output directory for FLUX.1-dev weights.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Output directory for FLUX.1-dev ControlNet adapter weights.",
    )
    parser.add_argument(
        "--download-synlayers",
        dest="download_synlayers",
        action="store_true",
        help="Download SynLayers ckpt folder into project root.",
    )
    parser.add_argument(
        "--synlayers-repo",
        dest="synlayers_repo",
        default=DEFAULT_SYNLAYERS_REPO,
        help="Hugging Face repo for SynLayers-compatible checkpoints.",
    )
    parser.add_argument(
        "--skip-flux",
        action="store_true",
        help="Skip downloading FLUX.1-dev weights.",
    )
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="Skip downloading FLUX.1-dev ControlNet adapter weights.",
    )
    return parser.parse_args()


def download_flux(target_dir):
    logger.info("Downloading FLUX.1-dev to %s", target_dir)
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )


def download_adapter(target_dir):
    logger.info("Downloading FLUX.1-dev-Controlnet-Inpainting-Alpha to %s", target_dir)
    snapshot_download(
        repo_id="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )


def download_synlayers_ckpt(project_root, repo_id):
    ckpt_dir = project_root / "ckpt"
    logger.info("Downloading SynLayers ckpt files into %s", ckpt_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(ckpt_dir),
        allow_patterns=[
            "decouple_LoRA/**",
            "pre_trained_LoRA/**",
            "prism_ft_LoRA/**",
            "trans_vae/**",
            "README.md",
        ],
        local_dir_use_symlinks=False,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    logger.info("Note: dataset JSONL sizes should be multiples of 8 for inference.")
    project_root = Path(args.project_root).resolve()
    default_ckpt_root = project_root / "ckpt"
    flux_dir = Path(args.flux_dir) if args.flux_dir else default_ckpt_root / "FLUX.1-dev"
    adapter_dir = (
        Path(args.adapter_dir)
        if args.adapter_dir
        else default_ckpt_root / "FLUX.1-dev-Controlnet-Inpainting-Alpha"
    )

    if not args.skip_flux:
        flux_dir.mkdir(parents=True, exist_ok=True)
        download_flux(flux_dir)
    if not args.skip_adapter:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        download_adapter(adapter_dir)
    if args.download_synlayers:
        download_synlayers_ckpt(project_root, args.synlayers_repo)


if __name__ == "__main__":
    main()


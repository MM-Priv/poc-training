"""TorchTitan launcher that registers FineWeb-Edu, then delegates to the
standard trainer entry point.

Uses the local HF cache (populated by training/download_data.py) in
non-streaming mode so training is not network-bound and the dataloader
has proper per-rank state_dict support for checkpoint resume.
"""

from functools import partial

from datasets import load_dataset
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.train import Trainer, main
# Side-effect import registers the Float8 / MX model converters in TorchTitan's
# registry so they are selectable via --model.converters quantize.linear.float8.
import torchtitan.components.quantization  # noqa: F401


def _load_fineweb_edu(path: str, split: str):
    # `path` comes from DatasetConfig.path (or --training.dataset_path override).
    # We reuse it as the HF datasets cache_dir so load_dataset hits local disk.
    return load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=split,
        streaming=False,
        cache_dir=path,
    )


def _extract_text(sample: dict) -> str:
    return sample["text"]


DATASETS["fineweb_edu"] = DatasetConfig(
    path="/mnt/data/poc-training/data/hf_cache",
    loader=partial(_load_fineweb_edu, split="train"),
    sample_processor=_extract_text,
)
DATASETS["fineweb_edu_validation"] = DatasetConfig(
    path="/mnt/data/poc-training/data/hf_cache",
    loader=partial(_load_fineweb_edu, split="train"),  # FineWeb-Edu ships only 'train'
    sample_processor=_extract_text,
)


if __name__ == "__main__":
    main(Trainer)

"""Dataset adapter for OakInk-v2 sequence-per-tar shards (egocentric GT ingestion).

Tars are produced by ``scripts/build/generate_oakink_world_res.py``: one
``OAKINK_<sanitized_seq_token>.tar`` per sequence containing the egocentric view as
``<clip_id>_f%05d.image.jpg`` (frames remapped to contiguous 0..T-1), with GT-derived
stage outputs in the matching ``seq_folder``.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

from lib.pipeline.datasets.base import AdapterValidationResult, BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor


_TAR_NAME_RE = re.compile(r"^(OAKINK_.+)\.tar$")
_IMAGE_MEMBER_RE = re.compile(r"^(OAKINK_.+)_f(\d+)\.image\.jpg$", re.IGNORECASE)


def _collect_tar_frames(tar_path: Path) -> tuple[list[str], list[list[int]]]:
    entries: list[tuple[int, str, list[int]]] = []
    with tarfile.open(tar_path, "r") as reader:
        for member in reader:
            if not member.isfile():
                continue
            match = _IMAGE_MEMBER_RE.match(Path(member.name).name)
            if match is None:
                continue
            entries.append((int(match.group(2)), member.name, [int(member.offset_data), int(member.size)]))
    if not entries:
        raise RuntimeError(f"No `*.image.jpg` frames found in {tar_path}")
    entries.sort(key=lambda item: item[0])
    indices = [item[0] for item in entries]
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise RuntimeError(f"Non-contiguous frame indices in {tar_path}: start={indices[0]} count={len(indices)}")
    return [item[1] for item in entries], [item[2] for item in entries]


@register_dataset_adapter
class OakInkTarDatasetAdapter(BaseDatasetAdapter):
    name = "oakink_tar"

    def build_descriptors(self, *, dataset_cfg: dict, adapter_cfg: dict, paths_cfg: dict, context=None, prepared=None):
        tar_root = Path(
            adapter_cfg.get("tar_root") or adapter_cfg.get("shard_dir")
            or paths_cfg.get("tar_root") or paths_cfg.get("shard_root", "")
        )
        if not tar_root.is_dir():
            raise FileNotFoundError(f"tar_root not found: {tar_root}")
        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (tar_root / "outputs"))

        descriptors = []
        for tar_path in sorted(tar_root.glob("OAKINK_*.tar")):
            clip_id = _TAR_NAME_RE.match(tar_path.name).group(1)
            frame_names, frame_offsets = _collect_tar_frames(tar_path)
            descriptors.append(
                ClipDescriptor.from_tar_shard(
                    clip_id=clip_id,
                    clip_name=clip_id,
                    root_dir=str(tar_root.resolve()),
                    seq_folder=str((seq_folder_root / clip_id).resolve()),
                    shard_path=str(tar_path.resolve()),
                    frame_names=frame_names,
                    frame_offsets=frame_offsets,
                    extra={"adapter": self.name, "dataset_name": dataset_cfg.get("source_id") or "oakink_v2"},
                )
            )
        return descriptors

    def validate_source(self, *, dataset_cfg: dict, adapter_cfg: dict, paths_cfg: dict, context=None, prepared=None, manifest_records=None):
        tar_root = Path(
            adapter_cfg.get("tar_root") or adapter_cfg.get("shard_dir")
            or paths_cfg.get("tar_root") or paths_cfg.get("shard_root", "")
        )
        if not tar_root.is_dir():
            return AdapterValidationResult(ok=False, summary={"adapter": self.name, "error": f"tar_root not found: {tar_root}"})
        tar_paths = sorted(tar_root.glob("OAKINK_*.tar"))
        if not tar_paths:
            return AdapterValidationResult(ok=False, summary={"adapter": self.name, "error": f"No OAKINK_*.tar in {tar_root}"})
        try:
            for tar_path in tar_paths[: max(1, int(adapter_cfg.get("validate_sample_shards", 1)))]:
                _collect_tar_frames(tar_path)
        except Exception as error:
            return AdapterValidationResult(ok=False, summary={"adapter": self.name, "error": str(error)})
        return AdapterValidationResult(ok=True, summary={"adapter": self.name, "tar_root": str(tar_root.resolve()), "tar_count": len(tar_paths)})

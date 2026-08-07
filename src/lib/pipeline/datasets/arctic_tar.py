"""Dataset adapter for ARCTIC ego sequence-per-tar shards (GT ingestion).

Tars are produced by ``scripts/build/generate_arctic_world_res.py``:
one ``ARCTIC_<subject>_<seq>.tar`` per ego sequence containing
``<clip_id>_f%05d.image.jpg`` frames (undistorted), with GT-derived stage outputs
in the matching ``seq_folder`` (world_space_res.pth, SLAM npz, infiller marker).
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

from lib.pipeline.datasets.base import AdapterValidationResult, BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor


_TAR_NAME_RE = re.compile(r"^(ARCTIC_(?P<subject>s\d{2})_(?P<seq>[A-Za-z0-9]+_[A-Za-z0-9]+(?:_\d+)?))\.tar$")
_IMAGE_MEMBER_RE = re.compile(r"^(ARCTIC_.+)_f(\d+)\.image\.jpg$", re.IGNORECASE)


def _parse_tar_identity(tar_path: Path) -> tuple[str, str, str]:
    match = _TAR_NAME_RE.match(tar_path.name)
    if match is None:
        raise ValueError(
            "ARCTIC tar filename must match `ARCTIC_<subject>_<seq>.tar`, "
            f"got: {tar_path.name}"
        )
    return match.group(1), match.group("subject"), match.group("seq")


def _collect_tar_frames(tar_path: Path) -> tuple[list[str], list[list[int]]]:
    image_entries: list[tuple[int, str, list[int]]] = []
    with tarfile.open(tar_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            image_match = _IMAGE_MEMBER_RE.match(Path(member.name).name)
            if image_match is None:
                continue
            frame_idx = int(image_match.group(2))
            image_entries.append((frame_idx, member.name, [int(member.offset_data), int(member.size)]))

    if not image_entries:
        raise RuntimeError(f"No `*.image.jpg` frames found in {tar_path}")

    image_entries.sort(key=lambda item: item[0])
    frame_indices = [item[0] for item in image_entries]
    expected_indices = list(range(frame_indices[0], frame_indices[0] + len(frame_indices)))
    if frame_indices != expected_indices:
        raise RuntimeError(
            f"Non-contiguous RGB frame indices in {tar_path}: "
            f"start={frame_indices[0]} count={len(frame_indices)}"
        )
    return [item[1] for item in image_entries], [item[2] for item in image_entries]


@register_dataset_adapter
class ARCTICTarDatasetAdapter(BaseDatasetAdapter):
    name = "arctic_tar"

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        tar_root = Path(
            adapter_cfg.get("tar_root")
            or adapter_cfg.get("shard_dir")
            or paths_cfg.get("tar_root")
            or paths_cfg.get("shard_root", "")
        )
        if not tar_root.is_dir():
            raise FileNotFoundError(f"tar_root not found: {tar_root}")

        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (tar_root / "outputs"))

        descriptors = []
        for tar_path in sorted(tar_root.glob("*.tar")):
            clip_id, subject, seq_name = _parse_tar_identity(tar_path)
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
                    extra={
                        "adapter": self.name,
                        "dataset_name": dataset_cfg.get("source_id") or "arctic",
                        "subject": subject,
                        "arctic_seq_name": seq_name,
                    },
                )
            )
        return descriptors

    def validate_source(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
        manifest_records=None,
    ) -> AdapterValidationResult:
        tar_root = Path(
            adapter_cfg.get("tar_root")
            or adapter_cfg.get("shard_dir")
            or paths_cfg.get("tar_root")
            or paths_cfg.get("shard_root", "")
        )
        if not tar_root.is_dir():
            return AdapterValidationResult(ok=False, summary={"adapter": self.name, "error": f"tar_root not found: {tar_root}"})

        tar_paths = sorted(tar_root.glob("*.tar"))
        if not tar_paths:
            return AdapterValidationResult(ok=False, summary={"adapter": self.name, "error": f"No tar shards found in {tar_root}"})

        sample = tar_paths[: max(1, int(adapter_cfg.get("validate_sample_shards", 1)))]
        try:
            for tar_path in sample:
                _parse_tar_identity(tar_path)
                _collect_tar_frames(tar_path)
        except Exception as error:
            return AdapterValidationResult(
                ok=False,
                summary={"adapter": self.name, "tar_root": str(tar_root.resolve()), "tar_count": len(tar_paths), "error": str(error)},
            )
        return AdapterValidationResult(
            ok=True,
            summary={"adapter": self.name, "tar_root": str(tar_root.resolve()), "tar_count": len(tar_paths), "sampled": len(sample)},
        )

# EgoSmith reconstruction image for the L4 fleet (detect_track,motion,slam,infiller).
#
# Built via Cloud Build (see cloudbuild.yaml); no local docker needed. The build
# REUSES the repo's authoritative installer `scripts/setup/setup_env.sh` so the env
# matches the dev box exactly — the ONLY divergence is TORCH_CUDA_ARCH_LIST=8.9, so
# DPVO's CUDA extensions target L4 (sm_89) instead of the dev box's H100 (sm_90).
# Verified locally: arch pin -> `arch = sm_89` cubins for cuda_ba/cuda_corr.
#
# Large checkpoints (hawor.ckpt 3.1G, infiller.pt 400M, any4d 7.2G) are NOT baked
# (see .dockerignore); pod_entry.sh pulls them from GCS into their resolver paths.
# Small assets (dpvo.pth, detector.pt, model_config.yaml, MANO _DATA/) ARE baked.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/conda/bin:/opt/google-cloud-sdk/bin:${PATH}

# build tools + ffmpeg/GL libs for cv2/decord (ninja is REQUIRED or DPVO falls back
# to a slow serial build).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget curl unzip ninja-build ca-certificates \
      ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Miniconda (setup_env.sh drives conda; it uses `conda run`, no shell activation).
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh

# Newer conda blocks non-interactive env creation until the default-channel ToS is
# accepted; accept it so setup_env.sh's `conda create` works headlessly.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# gcloud CLI — pod_entry.sh uses `gcloud storage` for GCS pull/push.
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt \
 && ln -sf /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud \
 && ln -sf /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil

WORKDIR /repo
COPY . /repo

# Build the egosmith conda env with the L4 arch pin. setup_env.sh is idempotent and
# builds DPVO via `pip install . --no-build-isolation`, which honors this env var.
RUN export TORCH_CUDA_ARCH_LIST=8.9 && bash scripts/setup/setup_env.sh

# AnyCalib (Apache-2.0) — learned camera intrinsics for --use_anycalib. --no-deps: torch
# is already pinned. Weights are pulled from GCS at pod init (pods may lack torch-hub net).
RUN conda run --no-capture-output -n egosmith pip install --no-deps "git+https://github.com/javrtg/AnyCalib.git"

# Reduced validation: prove DPVO imports (arch-correct link path) + numpy pin + the
# slam stage imports. Full weight checks are deferred to pod-init preflight (big
# ckpts are pulled at runtime). Single-line python -c (classic docker builder has no
# heredoc support).
RUN conda run --no-capture-output -n egosmith python -c "import numpy, dpvo, dpvo.lietorch, sys; sys.path.insert(0,'src'); import lib.pipeline.stages.slam; assert numpy.__version__=='1.26.4', numpy.__version__; print('[validate] dpvo+slam import OK; numpy', numpy.__version__)"

# Runtime scratch for HaWoR stage-3 (required by workspace/preflight). pod_entry.sh
# may override to a node-local SSD path.
ENV HAWOR_STAGE3_TMP_ROOT=/scratch/hawor_tmp \
    HAWOR_BATCH_TMPDIR=/scratch/hawor_batch_tmp

CMD ["bash", "scripts/fleet/egosmith_recon/pod_entry.sh"]

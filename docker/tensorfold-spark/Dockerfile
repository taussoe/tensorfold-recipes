# TensorFold for DGX Spark: NVIDIA's PyTorch container with one pinned TensorFold commit baked in.
#
# The recipes build this for you (./setup.sh), on both Sparks, as tensorfold-spark:<version>-<commit>.
# Baking the install into an image means a restart never re-installs, and both ranks provably run the same code.
FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG TENSORFOLD_COMMIT
RUN test -n "$TENSORFOLD_COMMIT" \
 && pip install --no-cache-dir "git+https://github.com/ashhart/TensorFold.git@${TENSORFOLD_COMMIT}" \
 && tensorfold --version

# No daily GitHub check from inside the cluster: the version is pinned on purpose.
# GB10 only: the container's default list (8.0 ... 12.0+PTX) makes torch build TensorFold's CUDA extensions for
# GPUs without FP8 MMA, which 0.3.5's prefill kernels need, and takes longer.
ENV TORCH_CUDA_ARCH_LIST=12.1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    PYTHONUNBUFFERED=1

# Compiled kernels. The recipes mount a host directory here, so the minute or two of kernel builds happens once
# per Spark instead of once per container.
ENV TORCH_EXTENSIONS_DIR=/kernel-cache/torch_extensions \
    TRITON_CACHE_DIR=/kernel-cache/triton

LABEL org.opencontainers.image.title="tensorfold-spark" \
      org.opencontainers.image.source="https://github.com/ashhart/TensorFold"

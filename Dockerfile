# syntax=docker/dockerfile:1

# Build:  docker build -t ninfer .
# Run:    docker run --gpus all -v /path/to/models:/models -p 8080:8080 ninfer
# The models folder must contain qwen3_8_27b.ninfer; for another artifact or flags, name the command:
#         docker run --gpus all -v /path/to/models:/models -p 8080:8080 ninfer \
#             ninfer-serve /models/other.ninfer --host 0.0.0.0 --max-context 32768
#
# Linux dev build and CTest (turing-dev preset; benchmarks are off unless --build-arg NINFER_BENCHMARKS=ON):
#         docker build --target dev -t ninfer-dev .
#         docker run --rm --gpus all --security-opt seccomp=unconfined -v /path/to/models:/models:ro ninfer-dev
# The read-queue tests need io_uring, which Docker's default seccomp profile blocks. The default command is
# ctest with the real-model tests skipped; a command such as `ctest --test-dir /build -R expert_cache`
# replaces it, and the real loading test takes the artifact on its command line:
#         ... ninfer-dev /build/tests/ninfer_qwen3_5_loading_real_test --artifact /models/qwen3_8_27b.ninfer

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS toolchain

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libhwloc-dev \
        liburing-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

FROM toolchain AS build

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DNINFER_TARGET=turing \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM toolchain AS dev

ARG DEBIAN_FRONTEND=noninteractive
ARG NINFER_BENCHMARKS=OFF
ARG NINFER_JOBS=4
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3 \
        python3-jinja2 \
        python3-jsonschema \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# nvcc runs out of memory at the default job count, so the build takes an explicit one.
RUN cmake --preset turing-dev -B /build -DNINFER_BUILD_BENCHMARKS=${NINFER_BENCHMARKS} \
    && cmake --build /build --parallel ${NINFER_JOBS}

CMD ["ctest", "--test-dir", "/build", "--output-on-failure"]

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libhwloc15 \
        liburing2 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

# The executables find the shared libraries through a relative RPATH (../lib), so both folders keep
# the build tree's layout; the PATH entries are symlinks, which the loader resolves.
COPY --from=build /build/apps/ninfer /build/apps/ninfer-serve /opt/ninfer/apps/
COPY --from=build /build/lib /opt/ninfer/lib
RUN ln -s /opt/ninfer/apps/ninfer /usr/local/bin/ninfer \
    && ln -s /opt/ninfer/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "/models/qwen3_8_27b.ninfer", "--host", "0.0.0.0"]

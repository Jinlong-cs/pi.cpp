FROM uniflexai/base_image:latest

ARG ARCH
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBERO_CONFIG_PATH=/root/.libero
ENV MUJOCO_GL=egl
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PYOPENGL_PLATFORM=egl
ENV UV_INDEX_URL=https://pypi.org/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang-format \
    cmake \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    ninja-build \
    pkg-config \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/*

RUN arch="${ARCH:-${TARGETARCH:-$(dpkg --print-architecture)}}"; \
    if [ "$arch" = "x86_64" ] || [ "$arch" = "amd64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
          libnvinfer10=10.13.0.35-1+cuda12.9 \
          libnvinfer-plugin10=10.13.0.35-1+cuda12.9 \
          libnvinfer-lean10=10.13.0.35-1+cuda12.9 \
          libnvinfer-vc-plugin10=10.13.0.35-1+cuda12.9 \
          libnvinfer-dispatch10=10.13.0.35-1+cuda12.9 \
          libnvonnxparsers10=10.13.0.35-1+cuda12.9 \
          libnvinfer-bin=10.13.0.35-1+cuda12.9 \
          libnvinfer-dev=10.13.0.35-1+cuda12.9 \
          libnvinfer-headers-dev=10.13.0.35-1+cuda12.9 \
          libnvinfer-plugin-dev=10.13.0.35-1+cuda12.9 \
          libnvinfer-headers-plugin-dev=10.13.0.35-1+cuda12.9 \
          libnvinfer-lean-dev=10.13.0.35-1+cuda12.9 \
          libnvinfer-vc-plugin-dev=10.13.0.35-1+cuda12.9 \
          libnvinfer-dispatch-dev=10.13.0.35-1+cuda12.9 \
          libnvonnxparsers-dev=10.13.0.35-1+cuda12.9 \
          python3-libnvinfer=10.13.0.35-1+cuda12.9 \
          python3-libnvinfer-dispatch=10.13.0.35-1+cuda12.9 \
          python3-libnvinfer-lean=10.13.0.35-1+cuda12.9 \
          python3-libnvinfer-dev=10.13.0.35-1+cuda12.9 \
      && rm -rf /var/lib/apt/lists/*; \
    elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
          tensorrt=10.3.0.30-1+cuda12.5 \
      && rm -rf /var/lib/apt/lists/*; \
    else \
      echo "Unsupported architecture: $arch"; exit 1; \
    fi

RUN python3 -m pip install --no-cache-dir --retries 10 --timeout 120 uv

ENV PI_CPP_HOME=/workspaces/pi.cpp
ENV TERM=xterm-256color
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}:/usr/src/tensorrt/bin"
ENV PYTHONPATH="${PYTHONPATH}:/opt/libero"

WORKDIR ${PI_CPP_HOME}

COPY . ${PI_CPP_HOME}

RUN python3 -m venv /opt/venv \
  && uv sync --frozen --no-dev --python /opt/venv/bin/python

RUN mkdir -p /root/.libero /opt/libero/libero/datasets \
  && printf '%s\n' \
    'benchmark_root: /opt/libero/libero/libero' \
    'bddl_files: /opt/libero/libero/libero/bddl_files' \
    'init_states: /opt/libero/libero/libero/init_files' \
    'datasets: /opt/libero/libero/datasets' \
    'assets: /opt/libero/libero/libero/assets' \
    > /root/.libero/config.yaml

EXPOSE 8000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["picpp-tui"]

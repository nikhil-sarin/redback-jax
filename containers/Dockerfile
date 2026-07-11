# CPU runtime image for redback-jax.
# Build (context = repo root): docker build -f containers/Dockerfile -t redback-jax:latest .
# GPU variant: containers/Dockerfile.gpu (jax[cuda12]).
FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y python3 python3-pip git build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

WORKDIR /src
COPY . /src

# JAX (CPU) first so a JAX problem is isolated from the package install.
RUN uv pip install --system --break-system-packages jax

# redback-jax core (pulls diffrax, jax-bandflux, unxt, wcosmo, ...).
# NOT the [inference]/[all] extra: it pins PyPI blackjax (>=1.0), a *different*
# package from the handley-lab nested-sampling fork that NestedSampler needs.
# Install the fork explicitly afterwards so it wins.
RUN uv pip install --system --break-system-packages .
RUN uv pip install --system --break-system-packages \
      "blackjax @ git+https://github.com/handley-lab/blackjax@nested_sampling" \
      anesthetic

# Fail the build if the runtime + nested-sampling API aren't importable.
RUN python3 -c "import redback_jax, jax, blackjax; from blackjax.ns.utils import log_weights; print('redback-jax + jax', jax.__version__, '+ blackjax(NS)', blackjax.__version__)"

CMD ["python3"]

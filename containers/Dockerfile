# CPU runtime image for redback-jax.
# Build (context = repo root): docker build -f containers/Dockerfile -t redback-jax:latest .
# GPU variant: containers/Dockerfile.gpu (jax[cuda12]).
FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y python3 python3-pip git build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

# External deps first (heavy wheels + a remote git checkout) so a local source
# change doesn't invalidate their cache layer. JAX (CPU), then the handley-lab
# nested-sampling blackjax fork the NestedSampler needs — pinned to a commit for
# reproducible builds — plus anesthetic. NOT the [inference]/[all] extra (below):
# it pins PyPI blackjax (>=1.0), a *different* package from this fork.
RUN uv pip install --system --break-system-packages jax
RUN uv pip install --system --break-system-packages \
      "blackjax @ git+https://github.com/handley-lab/blackjax@2180e29ffb645b2c46b76c768229c7c24212446c" \
      anesthetic

# redback-jax core last (pulls diffrax, jax-bandflux, unxt, wcosmo, ...).
WORKDIR /src
COPY . /src
RUN uv pip install --system --break-system-packages .

# Fail the build if the runtime + nested-sampling API aren't importable.
RUN python3 -c "import redback_jax, jax, blackjax; from blackjax.ns.utils import log_weights; print('redback-jax + jax', jax.__version__, '+ blackjax(NS)', blackjax.__version__)"

CMD ["python3"]

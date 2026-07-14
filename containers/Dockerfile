# CPU runtime image for redback-jax.
# Build (context = repo root): docker build -f containers/Dockerfile -t redback-jax:latest .
# GPU variant: containers/Dockerfile.gpu (jax[cuda12]).
FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y python3 python3-pip git build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

# External deps first (heavy wheels) so a local source change doesn't invalidate
# their cache layer. JAX (CPU), then mainline blackjax for the SMC+NUTS sampler
# used via run_nested_sampling. (The handley-lab nested-sampling fork is avoided:
# its older JAX API breaks on current JAX, and mainline SMC still gives evidence.)
RUN uv pip install --system --break-system-packages jax
RUN uv pip install --system --break-system-packages blackjax anesthetic

# redback-jax core last (pulls diffrax, jax-bandflux, unxt, wcosmo, ...).
WORKDIR /src
COPY . /src
RUN uv pip install --system --break-system-packages .

# Fail the build if the runtime + nested-sampling API aren't importable.
RUN python3 -c "import redback_jax, jax, blackjax; from redback_jax.inference.sampler import run_nested_sampling; print('redback-jax + jax', jax.__version__, '+ blackjax', blackjax.__version__)"

CMD ["python3"]

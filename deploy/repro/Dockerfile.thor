# Load the private recovery image first; build-thor-from-recovery.sh verifies its ID.
# Build on the matching aarch64 Thor host from this repository's root.
FROM spring-turret-demo:0.17.3-recovery-agxthor-5
COPY pyproject.toml README.md /tmp/turret-source/
COPY src /tmp/turret-source/src
RUN python -m pip install --no-deps --no-build-isolation /tmp/turret-source

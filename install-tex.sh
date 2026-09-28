#!/usr/bin/env bash
set -euo pipefail

# Install TeX Live into persistent home directory so it survives SageMaker restarts.
# Usage: bash install-tex.sh
# Optional: SCHEME env var overrides the scheme (default: scheme-basic)
#   e.g. SCHEME=scheme-full bash install-tex.sh

INSTALL_PREFIX="${HOME}/texlive"
SCHEME="${SCHEME:-scheme-basic}"

echo "==> Downloading install-tl..."
cd /tmp
wget -q --show-progress http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar -xzf install-tl-unx.tar.gz

echo "==> Installing TeX Live (scheme: ${SCHEME}) to ${INSTALL_PREFIX}..."
TEXLIVE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  /tmp/install-tl-*/install-tl --no-interaction --scheme="${SCHEME}"

# Locate the versioned bin directory (e.g. .../texlive/2026/bin/x86_64-linux)
TLBIN="$(echo "${INSTALL_PREFIX}"/*/bin/x86_64-linux)"

echo "==> Adding TeX Live to PATH in ~/.bashrc..."
# Avoid duplicate entries on re-runs
if ! grep -q "texlive" "${HOME}/.bashrc"; then
  cat >> "${HOME}/.bashrc" << EOF

# TeX Live (persistent install in home dir)
export PATH="${TLBIN}:\$PATH"
export MANPATH="${TLBIN}/../../../texmf-dist/doc/man:\$MANPATH"
export INFOPATH="${TLBIN}/../../../texmf-dist/doc/info:\$INFOPATH"
EOF
  echo "==> PATH updated in ~/.bashrc"
else
  echo "==> ~/.bashrc already contains a texlive entry, skipping."
fi

export PATH="${TLBIN}:${PATH}"

echo "==> Verifying installation..."
pdflatex --version

echo ""
echo "Done. Run 'source ~/.bashrc' to activate in the current session."
echo "To install additional packages: tlmgr install <package>"

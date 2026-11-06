#!/usr/bin/env bash
# Build dist/qadt_code.zip: everything Colab needs (backend, scripts, notebooks, built dashboard, docs).
# Upload it with notebooks/00_storage_setup.ipynb, or copy it to MyDrive/Quantum_Agentic_Digital_Twin_code/.
set -euo pipefail
cd "$(dirname "$0")/.."
(cd frontend && npm install --no-audit --no-fund >/dev/null && npx vite build >/dev/null)
mkdir -p dist
rm -f dist/qadt_code.zip
zip -qr dist/qadt_code.zip backend scripts notebooks documentation frontend/dist requirements.txt requirements-lock.txt README.md \
  -x "*/__pycache__/*" "*.pyc" ".dev_storage/*"
echo "dist/qadt_code.zip ($(du -h dist/qadt_code.zip | cut -f1))"

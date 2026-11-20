#!/usr/bin/env bash
# Download the NeRF-synthetic ("Blender") dataset used in the paper (8 scenes, ~800 MB zip)
# into data/nerf_synthetic/. The files are hosted on the authors' Google Drive; the same
# file id is used by nerfstudio's `ns-download-data blender`.
set -euo pipefail

mkdir -p data
cd data
if [ -d nerf_synthetic/lego ]; then
    echo "data/nerf_synthetic/lego already exists, nothing to do."
    exit 0
fi

pip install -q gdown
gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG -O nerf_synthetic.zip
# Unzip with Python, because many GPU images do not ship the `unzip` program.
python -c "import zipfile; zipfile.ZipFile('nerf_synthetic.zip').extractall('.')"
rm -f nerf_synthetic.zip
rm -rf __MACOSX
ls nerf_synthetic
echo "OK: data/nerf_synthetic/lego is ready."

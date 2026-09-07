conda create -n seavision-experiments python=3.12 -y
conda activate seavision-experiments
pip install open_clip_torch timm pandas numpy pillow matplotlib pyyaml tqdm requests jsonschema
python -m pip install torch>=2.9.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
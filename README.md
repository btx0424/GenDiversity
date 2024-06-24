# GenDiversity

## Installation

```bash
git clone --recursive git@github.com:btx0424/GenDiversity.git
# install GenSim
cd projects/GenSim && pip install -e .
# install diffusion_policy
cd ../diffusion_policy && pip install -e .
# install this repo
cd .. && pip install -e .
```

## Data Collection

For GenSim:

```bash
# at gen_diversity/scripts
python dataset/collect_gensim.py n=100 task=packing-boxes-pairs-seen-colors
```

## Training

DP for GenSim:
```bash
# at diffusion_policy/
bash run.sh
```
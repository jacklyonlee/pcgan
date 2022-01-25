# Point Cloud GAN
This repository implements an improved version of [Point Cloud GAN](https://github.com/chunliangli/Point-Cloud-GAN) (ICLR'19 Workshop) which achieves performance comparable to the [current SOTA](https://github.com/jw9730/setvae) in point cloud generation.

## Installation
* Set up and activate conda environment.

```shell
conda env create -f environment.yml
conda activate pcgan
```

* Compile CUDA extensions.

```shell
sh scripts/install.sh
```

* Download ShapeNet dataset and trained checkpoints.

```shell
sh scripts/download.sh
```

## Training
You can train using `train.py` or provided scripts.

```shell
# Train using CLI
python train.py --name NAME --cate airplane
# Train using provided settings
sh scripts/train_shapenet_aiplane.sh
```

## Testing
You can evaluate checkpointed models using `test.py` or provided scripts.

```shell
# Test user specified checkpoint using CLI
python test.py --ckpt_path CKPT_PATH --cate car
# Test provided checkpoints
sh scripts/test_shapenet_aiplane.sh
```

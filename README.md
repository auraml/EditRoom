# [ICLR 2025] EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing

Kaizhi Zheng, Xiaotong Chen, Xuehai He, Jing Gu, Linjie Li, Zhengyuan Yang, Kevin Lin, Jianfeng Wang, Lijuan Wang, Xin Eric Wang

[![arXiv](https://img.shields.io/badge/arXiv-2410.12836-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.12836)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://eric-ai-lab.github.io/edit-room.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

<p>
    <img width="730" alt="pipeline", src="./assets/new_teaser-1.png">
</p>

</h4>

This repository contains the official implementation of the paper: [EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing](https://arxiv.org/abs/2410.12836), which is accepted by ICLR 2025.
EditRoom is a graph diffusion-based generative model, which can manipulate 3D Room by using natural lanauge.

Feel free to contact me (kzheng31@ucsc.edu) or open an issue if you have any questions or suggestions.

## Getting Start

### 1. Installation
Clone our repo and create a new python environment.
```bash
git clone https://github.com/eric-ai-lab/EditRoom.git
cd EditRoom
conda create -n editroom python=3.10
conda activate editroom
pip install -r requirements.txt
```

Download the Blender software for visualization.
```bash
cd blender
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz
tar -xvf blender-3.3.1-linux-x64.tar.xz
rm blender-3.3.1-linux-x64.tar.xz
```

### 2. Creating Dataset

Dataset used in EditRoom is based on [3D-FORNT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) and [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future). Please refer to the instructions provided in their [official website](https://tianchi.aliyun.com/dataset/65347) to download the original dataset. Based on [InstructScene](https://huggingface.co/datasets/chenguolin/InstructScene_dataset), we write an automatic dataset generator to generate scene editing pairs. We provided preprocessed datasets on [HuggingFace](https://huggingface.co/datasets/KZ-ucsc/EditRoom_dataset/tree/main) for start.

First, downloading the preprocessed datasets. They will be downloaded under `datasets` folder by default. If you want to change to another directory, please remind to chage environment viariable `EDITROOM_DATA_FOLDER`.
```bash
export EDITROOM_DATA_FOLDER="./datasets"
PYTHONPATH=. python3 tools/download_dataset.py
```

**Note:** After executing this command, the following files will be automatically unzipped:
- `datasets/objfeat_vqvae.zip` will be unzipped to `datasets/data2/kzheng31/EditRoom_Public/objfeat_vqvae` 
- `datasets/preprocess.zip` will be unzipped to `datasets/data2/zhengkz/3D_datasets/preprocess`

However, both `objfeat_vqvae` and `preprocess` folders need to be directly under the `datasets` folder to comply with the required directory structure. Please run the following commands from the root directory to move them to the correct locations:

```bash
# Move objfeat_vqvae folder to datasets/
mv datasets/data2/kzheng31/EditRoom_Public/objfeat_vqvae datasets/
# Move preprocess folder to datasets/
mv datasets/data2/zhengkz/3D_datasets/preprocess datasets/

# Verify the folders are in the correct locations
ls -la datasets/objfeat_vqvae
ls -la datasets/preprocess

```

Then, please refer to [tools/README.md](./tools/README.md) for more details.

#### Required Directory Structure

After dataset preparation and setup, your directory structure should look like this:

```
EditRoom
├── assets
│   └── new_teaser-1.png
├── blender
│   └── blender-3.3.1-linux-x64
│       ├── 3.3
│       ├── blender
│       ├── blender.desktop
│       ├── blender-softwaregl
│       ├── blender.svg
│       ├── blender-symbolic.svg
│       ├── blender-thumbnailer
│       ├── copyright.txt
│       ├── lib
│       ├── license
│       └── readme.html
├── configs
│   ├── bedroom_sg2sc_diffusion.yaml
│   ├── bedroom_sg_diffusion.yaml
│   ├── bedroom_threed_front_splits.csv
│   ├── black_list.txt
│   ├── diningroom_sg2sc_diffusion.yaml
│   ├── diningroom_sg_diffusion.yaml
│   ├── diningroom_threed_front_splits.csv
│   ├── floor_plan_texture_images
│   │   └── floor_00003.jpg
│   ├── floor_plan_texture_images_references
│   ├── invalid_threed_front_rooms.txt
│   ├── livingroom_sg2sc_diffusion.yaml
│   ├── livingroom_sg_diffusion.yaml
│   └── livingroom_threed_front_splits.csv
├── constants.py
├── datasets
│   ├── 3D-FRONT
│   │   ├── 3D-FRONT
│   │   ├── 3D-FRONT-readme.md
│   │   ├── 3D-FRONT-texture
│   │   ├── 3D-FUTURE-model
│   │   └── threed_front.pkl
│   ├── 3D-FRONT.zip
│   ├── data2
│   │   ├── kzheng31
│   │   └── zhengkz
│   ├── editroom_dataset
│   │   ├── threed_front_bedroom
│   │   ├── threed_front_diningroom
│   │   └── threed_front_livingroom
│   ├── objfeat_vqvae
│   │   ├── objfeat_bounds.pkl
│   │   └── threedfront_objfeat_vqvae_epoch_01999.pth
│   ├── objfeat_vqvae.zip
│   ├── preprocess
│   │   ├── openshape_vitg14_indexs
│   │   ├── openshape_vitg14_indexs.tar
│   │   ├── openshape_vitg14_recon
│   │   ├── openshape_vitg14_recon.tar
│   │   ├── room_relations
│   │   └── room_relations.tar
│   └── preprocess.zip
├── extract_llm_plans.py
├── README.md
├── requirements.txt
├── run.sh
├── src
│   ├── data
│   ├── eval_edit.py
│   ├── infer.py
│   ├── models
│   ├── train_edit.py
│   └── utils
├── tools
└── weights (if you've downloaded ckpts and data pkl files from HF)
    ├── threedfront_bedroom_sg2sc_model-epoch=03-val_loss=0.171.ckpt
    ├── threedfront_bedroom_sg2sc_model-epoch=04-val_loss=0.146.ckpt
    ├── threedfront_bedroom_sg2sc_model-epoch=269-val_loss=0.018.ckpt
    ├── threedfront_bedroom_sg2sc_model-epoch=273-val_loss=0.018.ckpt
    ├── threedfront_bedroom_sg_model-epoch=213-val_loss=0.107.ckpt
    ├── threedfront_bedroom_sg_model-epoch=87-val_loss=0.111.ckpt
    ├── threedfront_bedroom_test_data.pkl
    ├── threed_front_bedroom_test_eval_tmp.pkl
    ├── threedfront_bedroom_train_data.pkl
```

### 3. Training

We will first train scene graph to scene layout generator:
```bash
PYTHONPATH=. python3 src/train_edit.py --config_file configs/bedroom_sg2sc_diffusion.yaml --output_directory ./weights --with_wandb_logger
```

Then, we will train scene graph generator:
```bash
PYTHONPATH=. python3 src/train_edit.py --config_file configs/bedroom_sg_diffusion.yaml --output_directory ./weights --with_wandb_logger
```

You can change `config_file` to other room types. All config files are under [configs](./configs/) folder.

### 4. Evaluation
To run the evaluation, we need both `SG_WEIGHT` and `SG2SC_WEIGHT`. Those weights should be found under `output_directory` for training.

If you runing on a headless server, please using `Xvfb` to create virtual screens for [pyrender](https://pyrender.readthedocs.io/en/latest/).
```bash
tmux new -s v_screen #Opening another terminal at the backend
sudo Xvfb :99 -screen 0 1960x1024x24
```

For evaluation, please run:
```bash
export DISPLAY=":99" #For headless server
PYTHONPATH=. python3 src/eval_edit.py --sg_config_file configs/bedroom_sg_diffusion.yaml \
        --sg2sc_config_file configs/bedroom_sg2sc_diffusion.yaml \
        --output_directory ./weights \
        --sg_weight_file SG_WEIGHT \
        --sg2sc_weight_file SG2SC_WEIGHT \
        --llm_plan_path LLM_PLAN
```

`LLM_PLAN` should be created during dataset generation. Please refer to [tools/README.md](./tools/README.md) for more details.

You can also using tempalte commands for evaluation.
```bash
export DISPLAY=":99" #For headless server
PYTHONPATH=. python3 src/eval_edit.py --sg_config_file configs/bedroom_sg_diffusion.yaml \
        --sg2sc_config_file configs/bedroom_sg2sc_diffusion.yaml \
        --output_directory ./weights \
        --sg_weight_file SG_WEIGHT \
        --sg2sc_weight_file SG2SC_WEIGHT
```

## Inference

### Checkpoint Download

Before running inference, you need to download the trained model checkpoints from our private Hugging Face repository.

First, login to Hugging Face CLI (this will prompt for your access token):
```bash
huggingface-cli login
```

Enter your Hugging Face access token when prompted. You can obtain an access token from [Hugging Face Settings](https://huggingface.co/settings/tokens) with read access to private repositories.

Then download the trained checkpoints:
```bash
python3 weights_download.py
```

This will download trained checkpoints from Hugging Face to the `./weights` directory for running inference.

### Running Inference

This guide shows how to use the `infer.py` script for interactive 3D scene editing. `infer.py` can be run directly after `weights_download.py` and edit pair creation by `edit_data_generator`

#### Script Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--source_scene_id` | Source scene UID (without .pkl extension) | `0d7be408-9e3d-4f68-8422-5aa2069ccdb2_MasterBedroom-27127` |
| `--room_type` | Room type (bedroom/livingroom/diningroom) | `bedroom` |
| `--sg_config_file` | Path to scene graph config file | `./configs/bedroom_sg_diffusion.yaml` |
| `--sg2sc_config_file` | Path to scene graph to scene config file | `./configs/bedroom_sg2sc_diffusion.yaml` |
| `--sg_weight_file` | Path to scene graph model weights | `/path/to/sg_model.ckpt` |
| `--sg2sc_weight_file` | Path to scene graph to scene model weights | `/path/to/sg2sc_model.ckpt` |
| `--output_directory` | Output directory for results | `single_scene_results` (default) |
| `--no_edit` | Only visualize original scene without editing | N/A (flag) |
| `--seed` | Random seed for reproducibility | `42` (default) |
| `--quiet` | Suppress verbose output and debug messages | N/A (flag) |

#### Usage Examples

##### 1. Visualize Original Scene Only

Use this mode to render and save the original scene without any editing:

```bash
PYTHONPATH=. python src/infer.py \
  --source_scene_id [SRC_SCENE_ID] \
  --room_type [bedroom/livingroom/diningroom] \
  --sg_config_file ./configs/bedroom_sg_diffusion.yaml \
  --sg2sc_config_file ./configs/bedroom_sg2sc_diffusion.yaml \
  --sg_weight_file [SG_WT_FILE_PATH] \
  --sg2sc_weight_file [SG2SC_WT_FILE_PATH] \
  --no_edit
```

##### 2. Interactive Scene Editing

Use this mode to edit scenes with natural language commands:

```bash
PYTHONPATH=. python src/infer.py \
  --source_scene_id [SRC_SCENE_ID] \
  --room_type [bedroom/livingroom/diningroom] \
  --sg_config_file ./configs/bedroom_sg_diffusion.yaml \
  --sg2sc_config_file ./configs/bedroom_sg2sc_diffusion.yaml \
  --sg_weight_file [SG_WT_FILE_PATH] \
  --sg2sc_weight_file [SG2SC_WT_FILE_PATH] \
```

The script will prompt you to enter editing commands like:
- "add a wardrobe next to the bed"
- "move the chair to the left"
- "rotate the desk 90 degrees"

##### 3. Quiet Mode (Suppressed Output)

Use this mode to reduce verbose output during editing:

```bash
PYTHONPATH=. python src/infer.py \
  --source_scene_id [SRC_SCENE_ID] \
  --room_type [bedroom/livingroom/diningroom] \
  --sg_config_file ./configs/bedroom_sg_diffusion.yaml \
  --sg2sc_config_file ./configs/bedroom_sg2sc_diffusion.yaml \
  --sg_weight_file [SG_WT_FILE_PATH] \
  --sg2sc_weight_file [SG2SC_WT_FILE_PATH] \
  --quiet
```

#### Output

Results are saved in timestamped folders under the output directory:
- **Original scene visualization**: `{output_dir}/{scene_id}_{timestamp}/original/`
- **Edited scene results**: `{output_dir}/{scene_id}_{timestamp}/generate/`
- **Multi-step editing**: `{output_dir}/{scene_id}_{timestamp}/edit_1/`, `edit_2/`, etc.

#### Requirements

- Valid OpenAI API key (set as `OPENAI_API_KEY` environment variable inside `constants.py`)
- Preprocessed 3D-FRONT dataset with VQ features
- Trained model weights for scene graph and scene generation
- Blender installation as per EditRoom author guidelines (for rendering)
- SRC_SCENE_ID must be present in either train or test folders under `./datasets/editroom_dataset/threed_front_{room_type}/`

## Acknowledgement
We would like to thank the authors of [ATISS](https://github.com/nv-tlabs/ATISS), [DiffuScene](https://github.com/tangjiapeng/DiffuScene), [OpenShape](https://github.com/Colin97/OpenShape_code), [NAP](https://arxiv.org/abs/2305.16315), [CLIPLayout](https://arxiv.org/abs/2303.03565) and [InstructScene](https://arxiv.org/abs/2402.04717) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.


## Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{
zheng2025editroom,
title={EditRoom: {LLM}-parameterized Graph Diffusion for Composable 3D Room Layout Editing},
author={Kaizhi Zheng and Xiaotong Chen and Xuehai He and Jing Gu and Linjie Li and Zhengyuan Yang and Kevin Lin and Jianfeng Wang and Lijuan Wang and Xin Eric Wang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```

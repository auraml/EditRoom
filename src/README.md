# EditRoom Single Scene Editing

This guide shows how to use the `single_scene_edit.py` script for interactive 3D scene editing.

## Setup

Navigate to the EditRoom directory in Remote Linux PC:
```bash
cd /home/ubuntu/Development/EditRoom/
```

## Script Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--source_scene_id` | Source scene UID (without .pkl extension) | `0d7be408-9e3d-4f68-8422-5aa2069ccdb2_MasterBedroom-27127` |
| `--room_type` | Room type (bedroom/livingroom/diningroom) | `bedroom` |
| `--sg_config_file` | Path to scene graph config file | `./configs/bedroom_sg_diffusion.yaml` |
| `--sg2sc_config_file` | Path to scene graph to scene config file | `./configs/bedroom_sg2sc_diffusion.ya|
| `--sg_weight_file` | Path to scene graph model weights | `/path/to/sg_model.ckpt` |
| `--sg2sc_weight_file` | Path to scene graph to scene model weights | `/path/to/sg2sc_model.ckpt` |
| `--output_directory` | Output directory for results | `single_scene_results` (default) |
| `--no_edit` | Only visualize original scene without editing | N/A (flag) |
| `--seed` | Random seed for reproducibility | `42` (default) |
| `--quiet` | Suppress verbose output and debug messages | N/A (flag) |

## Usage Examples

### 1. Visualize Original Scene Only

Use this mode to render and save the original scene without any editing:

```bash
PYTHONPATH=. python src/single_scene_edit.py \
  --source_scene_id 0d7be408-9e3d-4f68-8422-5aa2069ccdb2_MasterBedroom-27127 \
  --room_type bedroom \
  --sg_config_file ./configs/bedroom_sg_diffusion.yaml \
  --sg2sc_config_file ./configs/bedroom_sg2sc_diffusion.yaml \
  --sg_weight_file /home/ubuntu/Development/EditRoom/weights/threedfront_bedroom_sg_model-epoch=213-val_loss=0.107.ckpt \
  --sg2sc_weight_file /home/ubuntu/Development/EditRoom/weights/threedfront_bedroom_sg2sc_model-epoch=273-val_loss=0.018.ckpt \
  --no_edit
```

### 2. Interactive Scene Editing

Use this mode to edit scenes with natural language commands:

```bash
PYTHONPATH=. python src/single_scene_edit.py \
  --source_scene_id 0d7be408-9e3d-4f68-8422-5aa2069ccdb2_MasterBedroom-27127 \
  --room_type bedroom \
  --sg_config_file ./configs/bedroom_sg_diffusion.yaml \
  --sg2sc_config_file ./configs/bedroom_sg2sc_diffusion.yaml \
  --sg_weight_file /home/ubuntu/Development/EditRoom/weights/threedfront_bedroom_sg_model-epoch=213-val_loss=0.107.ckpt \
  --sg2sc_weight_file /home/ubuntu/Development/EditRoom/weights/threedfront_bedroom_sg2sc_model-epoch=273-val_loss=0.018.ckpt
```

The script will prompt you to enter editing commands like:
- "add a wardrobe next to the bed"
- "move the chair to the left"
- "rotate the desk 90 degrees"

### 3. Quiet Mode (Suppressed Output)

Use this mode to reduce verbose output during editing:

```bash
PYTHONPATH=. python src/single_scene_edit.py \
  --source_scene_id 0d7be408-9e3d-4f68-8422-5aa2069ccdb2_MasterBedroom-27127 \
  --room_type bedroom \
  --sg_config_file ./configs/bedroom_sg_diffusion.yaml \
  --sg2sc_config_file ./configs/bedroom_sg2sc_diffusion.yaml \
  --sg_weight_file /home/ubuntu/Development/EditRoom/weights/threedfront_bedroom_sg_model-epoch=213-val_loss=0.107.ckpt \
  --sg2sc_weight_file /home/ubuntu/Development/EditRoom/weights/threedfront_bedroom_sg2sc_model-epoch=273-val_loss=0.018.ckpt \
  --quiet
```

## Output

Results are saved in timestamped folders under the output directory:
- **Original scene visualization**: `{output_dir}/{scene_id}_{timestamp}/original/`
- **Edited scene results**: `{output_dir}/{scene_id}_{timestamp}/generate/`
- **Multi-step editing**: `{output_dir}/{scene_id}_{timestamp}/edit_1/`, `edit_2/`, etc.

## Requirements (Already setup in Remote Linux PC)

- Valid OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Preprocessed 3D-FRONT dataset with VQ features
- Trained model weights for scene graph and scene generation
- Blender installation as per EditRoom author guidelines (for rendering)
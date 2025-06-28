"""
Utility functions for single scene inference.
"""

# Standard library imports
import argparse
import ast
import datetime
import json
import os
import re
import requests
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import yaml

# Optional imports with fallbacks
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# Constants for magic numbers
MAX_FILE_MATCHES_DISPLAY = 5
API_RETRY_COUNT = 3
MAX_TOKENS = 2048
ANGLE_THRESHOLD_OBVIOUS = 135
ANGLE_THRESHOLD_SLIGHT = 45
SCALE_THRESHOLD_OBVIOUS_UP = 1.3
SCALE_THRESHOLD_OBVIOUS_DOWN = 0.7
COORDINATE_NORMALIZATION_THRESHOLD = 1.0
DISTANCE_THRESHOLD_OBVIOUS = 1.0
DISTANCE_THRESHOLD_SLIGHT = 0.5
STRING_PREVIEW_LENGTH = 100

print("Creating inference_utils.py")

def suppress_output():
    """Context manager to suppress stdout and stderr output."""
    null_file = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = null_file
        sys.stderr = null_file
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null_file.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Single scene editing with EditRoom")
    parser.add_argument("--source_scene_id", required=True,
                       help="Source scene UID (without .pkl extension)")
    parser.add_argument("--room_type", required=True,
                       choices=["bedroom", "livingroom", "diningroom"],
                       help="Room type")
    parser.add_argument("--sg_config_file", required=True,
                       help="Path to scene graph config file")
    parser.add_argument("--sg2sc_config_file", required=True,
                       help="Path to scene graph to scene config file")
    parser.add_argument("--sg_weight_file", required=True,
                       help="Path to scene graph model weights")
    parser.add_argument("--sg2sc_weight_file", required=True,
                       help="Path to scene graph to scene model weights")
    parser.add_argument("--output_directory", default="single_scene_results",
                       help="Output directory for results")
    parser.add_argument("--no_edit", action="store_true",
                       help="Only visualize original scene")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output and debug messages")

    return parser.parse_args()


def load_configs(sg_config_file: str, sg2sc_config_file: str, verbose: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load configuration files.

    Args:
        sg_config_file: Path to scene graph config file
        sg2sc_config_file: Path to scene graph to scene config file
        verbose: Whether to print loading message

    Returns:
        Tuple of (sg_config, sg2sc_config) dictionaries
    """
    if verbose:
        print("Loading configuration files...")

    # Load scene graph config
    with open(sg_config_file, "r") as f:
        sg_config = yaml.load(f, Loader=Loader)

    # Load scene graph to scene config
    with open(sg2sc_config_file, "r") as f:
        sg2sc_config = yaml.load(f, Loader=Loader)

    return sg_config, sg2sc_config


def validate_feature_paths(editroom_data_folder: str) -> None:
    """Validate that required feature extraction directories exist.

    Args:
        editroom_data_folder: Path to EDITROOM_DATA_FOLDER

    Raises:
        FileNotFoundError: If required directories are missing
    """
    print("Validating feature extraction paths...")

    paths_to_check = [
        os.path.join(editroom_data_folder, "preprocess", "openshape_vitg14_indexs"),
        os.path.join(editroom_data_folder, "preprocess", "openshape_vitg14_recon"),
        os.path.join(editroom_data_folder, "3D-FRONT", "3D-FUTURE-model")
    ]

    missing_paths = []
    for path in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(path)
        else:
            print(f"Found: {path}")

    if missing_paths:
        print("Missing required feature directories:")
        for path in missing_paths:
            print(f"  - {path}")
        raise FileNotFoundError(f"Required feature directories not found. Please complete preprocessing first.")

    print("All feature directories validated successfully")


def extract_obj_features(bbox):
    """
    Extract object features from a bounding box object.

    Args:
        bbox: Bounding box object with feature methods

    Returns:
        dict: Dictionary with extracted features

    Raises:
        ValueError: If any features cannot be extracted
    """
    features = {}

    # Get VQ reconstructed features
    try:
        if hasattr(bbox, 'openshape_vitg14_recon') and callable(bbox.openshape_vitg14_recon):
            features['objfeat_vq_recon'] = bbox.openshape_vitg14_recon()
            print(f"Loaded VQ recon features for {bbox.model_jid}: shape {features['objfeat_vq_recon'].shape}")
        else:
            raise ValueError(f"VQ recon method not available for {bbox.model_jid}")
    except Exception as e:
        print(f"Failed to load VQ recon for {bbox.model_jid}: {e}")
        raise ValueError(f"Cannot proceed without real VQ features for {bbox.model_jid}")

    # Get VQ indices
    try:
        if hasattr(bbox, 'openshape_vitg14_index') and callable(bbox.openshape_vitg14_index):
            features['objfeat_vq_indices'] = bbox.openshape_vitg14_index()
            print(f"Loaded VQ indices for {bbox.model_jid}: shape {features['objfeat_vq_indices'].shape}")
        else:
            raise ValueError(f"VQ index method not available for {bbox.model_jid}")
    except Exception as e:
        print(f"Failed to load VQ indices for {bbox.model_jid}: {e}")
        raise ValueError(f"Cannot proceed without real VQ indices for {bbox.model_jid}")

    # Get original features
    try:
        if hasattr(bbox, 'openshape_vitg14_features') and bbox.openshape_vitg14_features is not None:
            features['objfeat_vitg14_features'] = bbox.openshape_vitg14_features
            print(f"Loaded original features for {bbox.model_jid}: shape {features['objfeat_vitg14_features'].shape}")
        else:
            raise ValueError(f"Original features not available for {bbox.model_jid}")
    except Exception as e:
        print(f"Failed to load original features for {bbox.model_jid}: {e}")
        raise ValueError(f"Cannot proceed without real original features for {bbox.model_jid}")

    return features


def create_output_folder(output_directory: str, source_scene_id: str) -> str:
    """Creates output folder with timestamp.

    Args:
        output_directory: Base output directory
        source_scene_id: Scene ID to include in folder name

    Returns:
        Path to created output folder
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{source_scene_id}_{timestamp}"
    output_folder = os.path.join(output_directory, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def construct_plan_prompt(source_scene, instruction, class_labels, use_image=False):
    """
    Construct LLM prompt for planning.

    Args:
        source_scene: Source scene data
        instruction: Natural language instruction
        class_labels: List of available class labels
        use_image: Whether to include image (not supported in this version)

    Returns:
        dict: Message structure for LLM API
    """
    scene_description = source_scene.get_room_description()

    system_prompt = "Imagine you are a indoor room designer and you are using provided API to control the 3D models in the scene.\n" + \
        "Given one scene configuration and a command to edit the scene, you should use the provided APIs to do planning and achieve the target.\n" + \
        "All sizes and centroids in scene configurations are in meters. The angles are defined in degrees. The dimension sequence is [x,y,z]. Vertical angles are the angles along the y-axis.\n"+ \
        "Sizes are the half lengths of the bounding box along the x, y, and z axes when the vertical angle is zero.\n" + \
        "We define +x/-x as the right/left direction, +y/-y as the up/down direction, and +z/-z as the front/back direction.\n" + \
        "Positive angles are counterclockwise, and negative angles are clockwise.\n\n" + \
        "APIs:\n" + \
        "1. Rotate an object: ['Rotate', Target Object Description, Angle :(degrees)]\n" + \
        "2. Translate an object: ['Translate', Target Object Description, Direction :(x/z), Distance :(meters)]\n" + \
        "3. Scale an object: ['Scale', Target Object Description, Scale Factor]\n" + \
        "4. Replace an object: ['Replace', Source Object Description, Target Object Description]\n" + \
        "5. Add an object: ['Add', Target Object Description, (Relative Description, Relative Object Description)]\n" + \
        "6. Remove an object: ['Remove', Target Object Description]\n\n" + \
        "Matters needing attention:\n" + \
        "1. If there are multiple same objects in the scene and the command is related to the object, you should refer to the object locations.\n" + \
        "When you refer to the object locations, you should this format: (Relative Description, Relative Object Description). All reference should be append in the end of API lists.\n" + \
        "When you use add or remove command, you should refer to the object locations.\n" + \
        "Relative Description: [left, right, in front of, behind, above, below, closely left, closely right, closely in front of, closely behind]. 'closely' means the distance between two object centroids are less than 1 meters in x-z plane.\n" + \
        "For example, if you want to add a chair in front of the table, you should use the format: ['Add', 'chair', ('in front of', 'table')].\n" + \
        "At most add one relative description and one relative object description. Select the cloest one if there are more than two relative descriptions.\n" + \
        "The relative object description should be the same as the object description in the scene configuration.\n" + \
        "2. Translate, rotate, and scale commands should be executed in the order of scale, rotate, and translate.\n" + \
        "Translate should only work in the x/z direction. The distance should be one float number.\n" + \
        "3. When you scale an object, the object should be scaled uniformly. Scale factor should one float number.\n" + \
        "4. Replace object will only replace the object with the same class. Replace command will only change the object appearance, not the object poses and sizes.\n" + \
        "5. If Translate/Rotate/Scale commands can achieve the target, you should not use Replace/Add/Remove commands.\n" + \
        "6. If image is provided, you should use the image to help you understand the scene.\n" + \
        "7. Attempt to use the minimum number of commands to achieve the target.\n" + \
        f"8. If you want to add or replace object, you can only consider from these object classes: {json.dumps(class_labels)}.\n" + \
        "9. If you want to remove and add the object within the same class, you should use the replace command.\n" + \
        "10. Object descriptions should be detailed descriptions instead of class names. You can imagine the object descriptions if the object is not in the scene.\n" + \
        "11. Do not repeat the same API with the same objects.\n"+\
        "12. All apis should be able to converted to a list of strings and numbers, which can be directly processed by json.loads()\n\n"+\
        "For example:\n" + \
        "1. If you want to rotate a chair 90 degrees and there is only one chair in the scene, you should use the format: ['Rotate', 'chair', 90].\n" + \
        "2. If you want to add a chair in front of the wooden table, you should use the format: ['Add', 'chair', ('in front of', 'a wooden table')].\n" + \
        "3. If you want to remove a chair, you should use the format: ['Remove', 'chair'].\n" + \
        "4. If you want to replace a metal chair with a wooden one and this chair on the left of the bed with wooden design, you should use the format: ['Replace', 'the chair is metal', 'the chair is wooden', ('left', 'the bed is wooden')].\n\n" + \
        "Think about it step by step. Summarize the used apis at the end by lines. The final output format should be ***[api 1, api 2, ...]***.\n"

    prompt = "[Scene configurations]:\n" + scene_description + "\n" + \
        "[Command]:" + json.dumps(instruction) + "\n\n" + \
        "If there are multiple relative descriptions for one API, you should select the closest one.\n" + \
        "Checkout at the end to make sure output the final plan in the format of ***[api 1, api 2, ...]***.\n"

    content = [{
        "type": "text",
        "text": prompt
    }]

    message = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": MAX_TOKENS
    }
    return message


def call_llm_api(message, openai_api_key, retries=API_RETRY_COUNT):
    """
    Call OpenAI API with retry logic.

    Args:
        message: API message structure
        openai_api_key: OpenAI API key
        retries: Number of retries

    Returns:
        str: API response content or None if failed
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    for i in range(retries):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                   headers=headers, json=message, timeout=60)
            response_data = response.json()

            if 'choices' in response_data:
                return response_data['choices'][0]['message']['content']
            else:
                print(f"API call failed: {response_data}")
                if i < retries - 1:
                    time.sleep(2**i)
        except Exception as e:
            print(f"API call error: {e}")
            if i < retries - 1:
                time.sleep(2**i)

    return None


def convert_single_plan(plan):
    """
    Converts single plan to instruction format.

    Args:
        plan: List representing a single plan/command

    Returns:
        str: Formatted instruction string

    Raises:
        ValueError: If plan format is invalid
    """
    def add_relative(relative):
        direction = relative[0]
        target = relative[1]
        if "right" in direction:
            relative = "right of"
        elif "left" in direction:
            relative = "left of"
        elif "front" in direction:
            relative = "in front of"
        else:
            relative = direction

        if "closely" in direction and not "closely" in relative:
            relative = "closely " + relative

        return f"location: ***{relative}*** {target}"

    if plan[0] == "add":
        assert len(plan) == 3, "The add plan should have 3 elements"
        target = plan[1]
        relative = plan[2]
        # Handle both tuple and list formats for relative descriptions
        if isinstance(relative, (list, tuple)) and len(relative) == 2:
            relative_des = add_relative(relative)
            instruction = f"add object: {target}; {relative_des}."
        else:
            instruction = f"add object: {target}."
    elif plan[0] == "remove":
        assert len(plan) in [2, 3], "The remove plan should have 2 or 3 elements"
        target = plan[1]
        if len(plan) == 3:
            relative = plan[2]
            # Handle both tuple and list formats for relative descriptions
            if isinstance(relative, (list, tuple)) and len(relative) == 2:
                relative_des = add_relative(relative)
                instruction = f"remove object: {target}; {relative_des}."
            else:
                instruction = f"remove object: {target}."
        else:
            instruction = f"remove object: {target}."
    elif plan[0] == "translate":
        assert len(plan) in [4, 5], "The translate plan should have 4 or 5 elements"
        target = plan[1]
        direction = plan[2]
        distance = plan[3]
        assert direction in ['x', 'z'], "The direction should be x or z"
        assert type(distance) in [int, float], "The distance should be a number"
        direction_dict = {
            "x": "left" if distance < 0 else "right",
            "z": "front" if distance < 0 else "back",
        }
        distance = abs(distance)
        instruction = f"move object towards the ***{direction_dict[direction]}*** direction for {distance:.2f} meters: {target}"
        if distance > DISTANCE_THRESHOLD_OBVIOUS:
            instruction = "obviously " + instruction
        elif distance < DISTANCE_THRESHOLD_SLIGHT:
            instruction = "slightly " + instruction

        if len(plan) == 5:
            relative = plan[4]
            if isinstance(relative, (list, tuple)) and len(relative) == 2:
                relative_des = add_relative(relative)
                instruction += f"; {relative_des}."
    elif plan[0] == "rotate":
        assert len(plan) in [3, 4], "The rotate plan should have 3 or 4 elements"
        target = plan[1]
        angle = plan[2]
        assert type(angle) in [int, float], "The angle should be a number"
        if abs(angle) >= ANGLE_THRESHOLD_OBVIOUS:
            instruction = f"obviously rotate object {angle:.0f} degrees: {target}"
        elif abs(angle) <= ANGLE_THRESHOLD_SLIGHT:
            instruction = f"slightly rotate object {angle:.0f} degrees: {target}"
        else:
            instruction = f"rotate object {angle:.0f} degrees: {target}"
        if len(plan) == 4:
            relative = plan[3]
            if isinstance(relative, (list, tuple)) and len(relative) == 2:
                relative_des = add_relative(relative)
                instruction += f"; {relative_des}."
    elif plan[0] == 'scale':
        assert len(plan) in [3, 4], "The scale plan should have 3 or 4 elements"
        target = plan[1]
        scale = plan[2]
        assert type(scale) in [int, float], "The scale should be a number"
        if scale > 1:
            instruction = f"enlarge object by {scale:.1f} X: {target}"
            if scale > SCALE_THRESHOLD_OBVIOUS_UP:
                instruction = "obviously " + instruction
        elif scale < 1:
            instruction = f"shrink object by {scale:.1f} X: {target}"
            if scale < SCALE_THRESHOLD_OBVIOUS_DOWN:
                instruction = "obviously " + instruction
        else:
            instruction = None
        if len(plan) == 4:
            relative = plan[3]
            if isinstance(relative, (list, tuple)) and len(relative) == 2:
                relative_des = add_relative(relative)
                instruction += f"; {relative_des}."
    elif plan[0] == 'replace':
        assert len(plan) in [3, 4], "The replace plan should have 3 or 4 elements"
        source = plan[1]
        target = plan[2]
        instruction = f"replace source with target : [Source] {source}; [Target] {target}"
        if len(plan) == 4:
            relative = plan[3]
            if isinstance(relative, (list, tuple)) and len(relative) == 2:
                relative_des = add_relative(relative)
                instruction += f"; {relative_des}."
    else:
        raise ValueError(f"Invalid plan action: {plan}")

    assert instruction is not None, "Cannot process the instruction. Please check the plan."
    if instruction[-1] == "." and instruction[-2] == ".":
        instruction = instruction[:-1]
    return instruction


def extract_commands(response):
    """
    Extract and convert API commands from OpenAI response.
    Enhanced to handle nested JSON structures and multiple commands.

    Args:
        response: Raw LLM response text

    Returns:
        list: List of processed command instructions

    Raises:
        ValueError: If commands cannot be extracted
    """
    print(f"Extracting commands from response...")

    # First try to find JSON code blocks
    json_block_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_block_pattern, response, re.DOTALL)

    if json_match:
        json_text = json_match.group(1).strip()
        print(f"Found JSON code block")
        try:
            commands = json.loads(json_text)
            print(f"Successfully parsed JSON: {commands}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            json_match = None

    if not json_match:
        # Enhanced patterns to handle nested JSON structures
        patterns = [
            r'\*\*\*(\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\])\*\*\*',  # ***[nested arrays]***
            r'\*\*\*(\[.*?\])\*\*\*',  # ***[simple array]***
            r'(\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\])',  # Nested arrays anywhere
            r'(\[[^\[\]]*(?:\([^)]*\)[^\[\]]*)*\])',  # Simple arrays with parentheses
            r'\[(.*?)\]',  # Simple bracket matching (fallback)
        ]

        command_text = None
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                print(f"Found match with pattern {i}: {pattern}")
                # Take the longest match (most likely to be complete)
                command_text = max(matches, key=len)
                print(f"Selected match: {command_text[:STRING_PREVIEW_LENGTH]}...")
                break

        if not command_text:
            # Look for action-based commands as fallback
            action_pattern = r'((?:Remove|Add|Replace|Rotate|Translate|Scale)[^.]*\.)'
            action_matches = re.findall(action_pattern, response, re.MULTILINE | re.IGNORECASE)
            if action_matches:
                print(f"Found action-based commands: {action_matches}")
                # Convert to simple command format
                command_text = f"['{action_matches[0].split()[0]}', '{' '.join(action_matches[0].split()[1:])}']"
            else:
                raise ValueError(f"Could not extract commands from response")

        print(f"Extracted command text: {command_text}")

        # Parse the extracted command text
        try:
            # First try direct JSON parsing
            try:
                commands = json.loads(command_text)
                print(f"Direct JSON parsing successful")
            except json.JSONDecodeError:
                # Try AST parsing
                commands = ast.literal_eval(command_text)
                print(f"AST parsing successful")

        except (ValueError, SyntaxError) as e:
            print(f"Standard parsing failed: {e}")

            # Try to fix common JSON issues
            fixed_text = command_text
            # Fix single quotes to double quotes
            fixed_text = re.sub(r"'([^']*)'", r'"\1"', fixed_text)
            # Fix parentheses tuples to arrays
            fixed_text = re.sub(r'\(([^)]*)\)', r'[\1]', fixed_text)

            try:
                commands = json.loads(fixed_text)
                print(f"Fixed JSON parsing successful: {commands}")
            except json.JSONDecodeError:
                print("All parsing methods failed, trying manual extraction...")
                raise ValueError("Could not parse command structure")

    # Handle the case where AST parsing returns a tuple of commands
    if isinstance(commands, tuple) and len(commands) >= 2:
        # Check if this is a tuple of individual commands
        if all(isinstance(cmd, (list, tuple)) and len(cmd) >= 2 for cmd in commands):
            # This is a tuple of commands, convert to list
            commands = [list(cmd) for cmd in commands]
        else:
            # This is a single command in tuple format
            commands = [list(commands)]
    elif not isinstance(commands, list):
        commands = [commands]

    # Ensure all commands are in list format
    processed_commands = []
    for i, cmd in enumerate(commands):
        if isinstance(cmd, (list, tuple)):
            processed_commands.append(list(cmd))
        else:
            processed_commands.append([cmd])

    commands = processed_commands

    # Convert to instructions
    instructions = []
    for cmd in commands:
        if isinstance(cmd, (list, tuple)) and len(cmd) >= 2:
            cmd_list = list(cmd)
            # Convert first element to lowercase for consistency
            cmd_list[0] = str(cmd_list[0]).lower()
            print(f"Processing command: {cmd_list}")
            try:
                instruction = convert_single_plan(cmd_list)
                instructions.append(instruction)
                print(f"Converted to instruction: {instruction}")
            except Exception as e:
                print(f"Error converting command {cmd_list}: {e}")
                # Continue with other commands instead of failing completely
                continue
        else:
            print(f"Skipping invalid command (insufficient elements): {cmd}")

    if not instructions:
        raise ValueError("No valid commands could be processed")

    return instructions


def validate_coordinates(scene_params: Dict[str, Any], raw_dataset, verbose: bool = True) -> Dict[str, Any]:
    """Validate and denormalize scene coordinates if needed.

    Args:
        scene_params: Scene parameters with translations, sizes, etc.
        raw_dataset: Dataset with normalization bounds
        verbose: Whether to print debug information

    Returns:
        Fixed scene parameters with denormalized coordinates
    """
    translations = scene_params['translations']
    sizes = scene_params['sizes']

    if verbose:
        print(f"Validating coordinates...")
        print(f"Translation range: [{np.min(translations):.3f}, {np.max(translations):.3f}]")
        print(f"Size range: [{np.min(sizes):.3f}, {np.max(sizes):.3f}]")

    if (np.all(np.abs(translations) <= COORDINATE_NORMALIZATION_THRESHOLD) and
            np.all(sizes <= COORDINATE_NORMALIZATION_THRESHOLD)):
        print("WARNING: Detected normalized coordinates, denormalizing...")

        if hasattr(raw_dataset, 'centroids') and hasattr(raw_dataset, 'sizes'):
            centroids_min, centroids_max = raw_dataset.centroids
            sizes_min, sizes_max = raw_dataset.sizes

            print(f"Dataset centroid bounds: [{centroids_min}, {centroids_max}]")
            print(f"Dataset size bounds: [{sizes_min}, {sizes_max}]")

            # Denormalize translations: from [-1,1] to [min,max]
            translations = translations * (centroids_max - centroids_min) / 2.0 + (centroids_max + centroids_min) / 2.0

            # Denormalize sizes: from [0,1] to [min,max]
            sizes = sizes * (sizes_max - sizes_min) + sizes_min

            scene_params['translations'] = translations
            scene_params['sizes'] = sizes

            print(f"Coordinates denormalized successfully")
            print(f"New translation range: [{np.min(translations):.3f}, {np.max(translations):.3f}]")
            print(f"New size range: [{np.min(sizes):.3f}, {np.max(sizes):.3f}]")
        else:
            print("Warning: Dataset bounds not available, cannot denormalize")
    else:
        print("Coordinates appear to be in real-world scale (meters)")

    return scene_params


def extract_base_scene_id(scene_id: str, verbose: bool = True) -> str:
    """Extract base scene ID by removing edit suffixes.

    Args:
        scene_id: Scene ID that may contain edit suffixes
        verbose: Whether to print debug information

    Returns:
        Base scene ID without edit suffixes
    """
    suffixes = ['_pose-', '_remove-', '_add-', '_replace-']
    base_id = scene_id

    for suffix in suffixes:
        if suffix in base_id:
            base_id = base_id.split(suffix)[0]
            break

    if verbose:
        print(f"Extracted base scene ID: '{scene_id}' -> '{base_id}'")
    return base_id


def find_base_scene(base_scene_id: str, dataset_folder: str, verbose: bool = True) -> str:
    """Find the base scene file in dataset folder.

    Args:
        base_scene_id: Base scene ID without edit suffixes
        dataset_folder: Path to dataset folder
        verbose: Whether to print debug information

    Returns:
        Full path to the base scene .pkl file

    Raises:
        FileNotFoundError: If base scene not found
    """
    base_scene_file = f"{base_scene_id}.pkl"
    base_scene_path = os.path.join(dataset_folder, base_scene_file)

    if os.path.exists(base_scene_path):
        if verbose:
            print(f"Found exact base scene match: {base_scene_file}")
        return base_scene_path

    if verbose:
        print(f"Exact match not found, searching for base scene pattern...")
    edit_suffixes = ['_pose-', '_remove-', '_add-', '_replace-']

    for filename in os.listdir(dataset_folder):
        if filename.startswith(base_scene_id) and filename.endswith('.pkl'):
            # Check if this file has no edit suffixes
            has_edit_suffix = any(suffix in filename for suffix in edit_suffixes)
            if not has_edit_suffix:
                full_path = os.path.join(dataset_folder, filename)
                if verbose:
                    print(f"Found base scene without edit suffixes: {filename}")
                return full_path

    matching_files = [f for f in os.listdir(dataset_folder) if f.startswith(base_scene_id) and f.endswith('.pkl')]
    if matching_files:
        if verbose:
            print(f"Found {len(matching_files)} files matching pattern:")
            for f in matching_files[:MAX_FILE_MATCHES_DISPLAY]:  # Show first 5 matches
                print(f"  - {f}")
        # Use the first match as fallback
        fallback_path = os.path.join(dataset_folder, matching_files[0])
        if verbose:
            print(f"Warning: Using fallback file: {matching_files[0]}")
        return fallback_path

    raise FileNotFoundError(f"Base scene {base_scene_id} not found in {dataset_folder}")
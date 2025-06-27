# Standard library imports
import argparse
import datetime
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import requests
import torch
import yaml
from lightning.pytorch import seed_everything
from tqdm import tqdm

# Optional imports with fallbacks
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

try:
    from shapely.geometry import Polygon, Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("COLLISION_DEBUG: Warning - Shapely not available, collision detection may be limited")

try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("COLLISION_DEBUG: Warning - SciPy not available, some collision functions may not work")

# Local imports
from constants import EDIT_DATA_FOLDER, EDITROOM_DATA_FOLDER, OPENAI_API_KEY, BLENDER_PATH
from src.data.threed_front import ThreedFront
from src.data.utils_data import get_dataset
from src.models.room_edit import RoomEdit
from src.utils.util import construct_scene_from_vq_objdata, get_blender_render, render_generated_scene
from src.utils.visualize import export_scene
# Import utility functions from inference_utils
from src.utils.inference_utils import (
    suppress_output, parse_arguments, load_configs, validate_feature_paths,
    extract_obj_features, create_output_folder, construct_plan_prompt,
    call_llm_api, convert_single_plan, extract_commands, validate_coordinates,
    extract_base_scene_id, find_base_scene
)

# Constants for magic numbers
MAX_COLLISION_DISPLAY = 3
MAX_FILE_MATCHES_DISPLAY = 5
DEFAULT_CAMERA_DISTANCE = 1.2
API_RETRY_COUNT = 3
MAX_TOKENS = 2048
ANGLE_THRESHOLD_OBVIOUS = 135
ANGLE_THRESHOLD_SLIGHT = 45
SCALE_THRESHOLD_OBVIOUS_UP = 1.3
SCALE_THRESHOLD_OBVIOUS_DOWN = 0.7
MAX_COLLISION_RESOLUTION_ATTEMPTS = 100
COORDINATE_NORMALIZATION_THRESHOLD = 1.0
DISTANCE_THRESHOLD_OBVIOUS = 1.0
DISTANCE_THRESHOLD_SLIGHT = 0.5
COLLISION_SEPARATION_FACTOR = 0.6
COLLISION_SAFETY_MARGIN = 0.1
COMMAND_PREVIEW_LENGTH = 50
STRING_PREVIEW_LENGTH = 100
DEFAULT_ZERO_ANGLE = 0.0

# COLLISION DETECTION DEBUGGING: Import collision functions from tools
tools_path = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.insert(0, tools_path)
try:
    # Import directly from tools.utils module
    import tools.utils as tools_utils

    # Ensure all dependencies are available in the tools_utils module
    tools_utils.np = np
    tools_utils.math = math
    tools_utils.deepcopy = deepcopy
    if SHAPELY_AVAILABLE:
        tools_utils.Polygon = Polygon
        tools_utils.Point = Point
    if SCIPY_AVAILABLE:
        tools_utils.R = R

    two_rectangle_collision = tools_utils.two_rectangle_collision
    check_collision = tools_utils.check_collision
    check_collision_all = tools_utils.check_collision_all
    COLLISION_DETECTION_AVAILABLE = True
    print("Collision detection enabled")
except ImportError as e:
    try:
        # Fallback: try importing the utils.py file directly
        utils_file_path = os.path.join(tools_path, 'utils.py')
        spec = importlib.util.spec_from_file_location("tools_utils", utils_file_path)
        tools_utils = importlib.util.module_from_spec(spec)

        # Inject dependencies before executing module
        tools_utils.np = np
        tools_utils.math = math
        tools_utils.deepcopy = deepcopy
        if SHAPELY_AVAILABLE:
            tools_utils.Polygon = Polygon
            tools_utils.Point = Point
        if SCIPY_AVAILABLE:
            tools_utils.R = R

        spec.loader.exec_module(tools_utils)

        two_rectangle_collision = tools_utils.two_rectangle_collision
        check_collision = tools_utils.check_collision
        check_collision_all = tools_utils.check_collision_all
        COLLISION_DETECTION_AVAILABLE = True
        print("Collision detection enabled")
    except Exception as e2:
        COLLISION_DETECTION_AVAILABLE = False
        print(f"COLLISION_DEBUG: Failed to import collision detection (both methods failed):")
        print(f"  Method 1 error: {e}")
        print(f"  Method 2 error: {e2}")
        print(f"  Tools path: {tools_path}")
        print(f"  Utils file exists: {os.path.exists(os.path.join(tools_path, 'utils.py'))}")
        print(f"  Available dependencies: shapely={SHAPELY_AVAILABLE}, scipy={SCIPY_AVAILABLE}")

def debug_check_scene_collisions(scene_data, step_description: str = "", verbose: bool = True) -> bool:
    """Check for collisions in a scene and print detailed information.

    Args:
        scene_data: Scene data object containing bboxes
        step_description: Description of current processing step
        verbose: Whether to print debug information

    Returns:
        True if collisions found, False otherwise
    """
    if not COLLISION_DETECTION_AVAILABLE:
        if verbose:
            print(f"COLLISION_DEBUG: [{step_description}] Collision detection unavailable")
        return False

    try:
        # Validate object properties first
        for i, obj in enumerate(scene_data.bboxes):
            if not hasattr(obj, 'position') or not hasattr(obj, 'size'):
                continue
            if not hasattr(obj, 'z_angle'):
                obj.z_angle = DEFAULT_ZERO_ANGLE

        # Check all pairs for collisions
        collision_found = check_collision_all(scene_data)
        collision_pairs = []

        if collision_found:
            # Find specific collision pairs
            for i in range(len(scene_data.bboxes)):
                for j in range(i + 1, len(scene_data.bboxes)):
                    try:
                        if two_rectangle_collision(scene_data.bboxes[i], scene_data.bboxes[j]):
                            obj1 = scene_data.bboxes[i]
                            obj2 = scene_data.bboxes[j]
                            collision_pairs.append((i, j, obj1, obj2))
                    except:
                        pass

        if verbose:
            if collision_found:
                print(f"[{step_description}] COLLISION DETECTED: "
              f"{len(collision_pairs)} collision(s) detected among {len(scene_data.bboxes)} objects")
                for i, j, obj1, obj2 in collision_pairs[:MAX_COLLISION_DISPLAY]:  # Show max 3 collisions
                    print(f"  - {obj1.label} ↔ {obj2.label}")
            else:
                print(f"[{step_description}] NO COLLISIONS: ({len(scene_data.bboxes)} objects)")

        return collision_found

    except Exception as e:
        if verbose:
            print(f"COLLISION_DEBUG: [{step_description}] Error checking collisions: {e}")
        return False

QUIET_MODE = False

def qprint(*args, **kwargs) -> None:
    """Helper function to suppress blender output in quiet mode.

    Args:
        *args: Arguments to pass to print()
        **kwargs: Keyword arguments to pass to print()
    """
    if not QUIET_MODE:
        print(*args, **kwargs)


# extract_base_scene_id removed - now imported from inference_utils


# find_base_scene removed - now imported from inference_utils


def ss_fallback(scene_id: str, room_type: str) -> Tuple[Any, Any, Any]:
    """Fallback function that uses the original scene loading method.

    Used when base scene detection fails.

    Args:
        scene_id: Scene ID to load
        room_type: Type of room (bedroom, livingroom, diningroom)

    Returns:
        Tuple of (scene_data, temp_raw_dataset, object_dataset)
    """
    qprint("Using fallback scene loading method...")

    # Create temporary dataset folder structure
    temp_dataset_folder = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}", "temp_single_scene_batch")
    os.makedirs(temp_dataset_folder, exist_ok=True)

    # Load the original test dataset to get the scene
    test_data_folder = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}", "test_dataset")
    data_stats_path = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}", "dataset_stats.txt")

    qprint("Loading test dataset to find scene...")
    with tqdm(desc="Loading test dataset", unit="scene") as pbar:
        raw_dataset = ThreedFront.load_from_folder(test_data_folder, path_to_train_stats=data_stats_path)
        pbar.update(1)

    # Find the specific scene using original method
    scene_data = None
    scene_file_path = None
    with tqdm(desc="Searching for scene", unit="scene") as pbar:
        for scene_path in raw_dataset.scenes:
            if isinstance(scene_path, str):
                scene_filename = os.path.basename(scene_path)
                if scene_id in scene_filename:
                    with open(scene_path, "rb") as f:
                        scene_data = pickle.load(f)
                    scene_file_path = scene_path
                    break
            pbar.update(1)

    if scene_data is None:
        raise ValueError(f"Scene {scene_id} not found in dataset")

    qprint(f"Found scene using fallback: {scene_data.uid}")

    # Extract base scene ID using old method
    base_scene_id = scene_id.split('_pose-')[0] if '_pose-' in scene_id else scene_id
    base_scene_id = base_scene_id.split('_remove-')[0] if '_remove-' in base_scene_id else base_scene_id
    base_scene_id = base_scene_id.split('_add-')[0] if '_add-' in base_scene_id else base_scene_id
    base_scene_id = base_scene_id.split('_replace-')[0] if '_replace-' in base_scene_id else base_scene_id

    scene_data.original_id = base_scene_id

    # Continue with rest of processing...
    scene_data.command = "dummy_command_for_single_scene_editing"

    # Copy the scene file to temporary folder
    temp_scene_path = os.path.join(temp_dataset_folder, os.path.basename(scene_file_path))

    # Save the modified scene data
    with open(temp_scene_path, "wb") as f:
        pickle.dump(scene_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create a temporary dataset with just this scene
    temp_raw_dataset = ThreedFront([temp_scene_path], path_to_train_stats=data_stats_path)
    temp_raw_dataset.uid_to_scene_index = {scene_data.uid: 0, base_scene_id: 0}

    # Load object dataset
    object_save_path = os.path.join(
        EDIT_DATA_FOLDER, f"threed_front_{room_type}", f"threed_front_{room_type}_objects.pkl"
    )
    with tqdm(desc="Loading object dataset", unit="step") as pbar:
        with open(object_save_path, "rb") as f:
            object_dataset = pickle.load(f)
        pbar.update(1)

    return scene_data, temp_raw_dataset, object_dataset


# parse_arguments removed - now imported from inference_utils


# load_configs removed - now imported from inference_utils


def load_single_scene(scene_id: str, room_type: str) -> Tuple[Any, Any, Any]:
    """Load single scene data with enhanced base scene detection.

    Now properly loads base scenes instead of pre-edited scenes.

    Args:
        scene_id: Scene ID to load
        room_type: Type of room (bedroom, livingroom, diningroom)

    Returns:
        Tuple of (scene_data, temp_raw_dataset, object_dataset)
    """
    qprint(f"Loading scene {scene_id} for room type {room_type}...")


    base_scene_id = extract_base_scene_id(scene_id, verbose=not QUIET_MODE)


    temp_dataset_folder = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}", "temp_single_scene_batch")
    os.makedirs(temp_dataset_folder, exist_ok=True)

    test_data_folder = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}", "test_dataset")
    train_data_folder = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}", "train_dataset")
    data_stats_path = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}", "dataset_stats.txt")

    qprint("Searching for base scene file...")
    base_scene_file_path = None
    dataset_folder_used = None

    if os.path.exists(test_data_folder):
        try:
            base_scene_file_path = find_base_scene(base_scene_id, test_data_folder, verbose=not QUIET_MODE)
            dataset_folder_used = test_data_folder
            qprint(f"Base scene found in test dataset: {os.path.basename(base_scene_file_path)}")
        except FileNotFoundError:
            qprint("Base scene not found in test dataset, checking train dataset...")

    if base_scene_file_path is None and os.path.exists(train_data_folder):
        try:
            base_scene_file_path = find_base_scene(base_scene_id, train_data_folder, verbose=not QUIET_MODE)
            dataset_folder_used = train_data_folder
            qprint(f"Base scene found in train dataset: {os.path.basename(base_scene_file_path)}")
        except FileNotFoundError:
            qprint("Base scene not found in train dataset either")


    if base_scene_file_path is None:
        qprint(f"Base scene search failed in both train and test datasets")
        qprint("Falling back to original search method...")

        return ss_fallback(scene_id, room_type)

    qprint("Loading base scene data...")
    with open(base_scene_file_path, "rb") as f:
        scene_data = pickle.load(f)

    qprint(f"Loaded base scene: {scene_data.uid}")

    if any(suffix in scene_data.uid for suffix in ['_pose-', '_remove-', '_add-', '_replace-']):
        qprint(f"Warning: Loaded scene {scene_data.uid} appears to have edit suffixes")
        qprint("This may indicate the base scene detection needs adjustment")

    scene_data.original_id = base_scene_id

    scene_data.command = "dummy_command_for_single_scene_editing"


    temp_scene_path = os.path.join(temp_dataset_folder, os.path.basename(base_scene_file_path))


    with open(temp_scene_path, "wb") as f:
        pickle.dump(scene_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Loading dataset statistics...")
    temp_raw_dataset = ThreedFront([temp_scene_path], path_to_train_stats=data_stats_path)
    temp_raw_dataset.uid_to_scene_index = {scene_data.uid: 0, base_scene_id: 0}


    object_save_path = os.path.join(
        EDIT_DATA_FOLDER, f"threed_front_{room_type}", f"threed_front_{room_type}_objects.pkl"
    )
    with tqdm(desc="Loading object dataset", unit="step") as pbar:
        with open(object_save_path, "rb") as f:
            object_dataset = pickle.load(f)
        pbar.update(1)

    print(f"Successfully loaded base scene from {'test' if 'test' in dataset_folder_used else 'train'} dataset")
    return scene_data, temp_raw_dataset, object_dataset


# validate_coordinates removed - now imported from inference_utils


# validate_feature_paths removed - now imported from inference_utils


# extract_obj_features removed - now imported from inference_utils


def convert_scene_to_params(scene_data, raw_dataset=None):
    """

    Args:
        scene_data: Scene data object
        raw_dataset: Dataset with class labels and bounds

    Returns:
        dict: Scene parameters with real object features
    """
    print("Converting scene to parameters with real feature extraction...")

    validate_feature_paths(EDITROOM_DATA_FOLDER)

    n_objects = len(scene_data.bboxes)
    print(f"Processing {n_objects} objects...")

    translations = np.array([bbox.position for bbox in scene_data.bboxes])
    sizes = np.array([bbox.size for bbox in scene_data.bboxes])
    angles = np.array([[bbox.z_angle] for bbox in scene_data.bboxes])  # Shape: (n_objects, 1)


    class_labels = [bbox.label for bbox in scene_data.bboxes]

    if raw_dataset is not None and hasattr(raw_dataset, 'class_labels'):
        try:
            class_labels_ids = [raw_dataset.class_labels.index(label) for label in class_labels]
        except ValueError as e:
            print(f"Warning: Some class labels not found in dataset: {e}")
            # Fallback to dummy indices
            class_labels_ids = list(range(n_objects))
    else:
        # Fallback to dummy indices
        class_labels_ids = list(range(n_objects))

    scene_params = {
        'translations': translations,
        'sizes': sizes,
        'angles': angles,
        'objs': class_labels_ids,
        'room_uid': scene_data.uid,
    }

    # PHASE 3 FIX: Extract REAL object features with NO fallbacks
    objfeat_vq_recon = []
    objfeat_vq_indices = []
    objfeat_vitg14_features = []

    for i, bbox in enumerate(scene_data.bboxes):
        print(f"Processing object {i+1}/{n_objects}: {bbox.label} (jid: {bbox.model_jid})")

        try:
            features = extract_obj_features(bbox)
            objfeat_vq_recon.append(features['objfeat_vq_recon'])
            objfeat_vq_indices.append(features['objfeat_vq_indices'])
            objfeat_vitg14_features.append(features['objfeat_vitg14_features'])

        except ValueError as e:
            print(f"{e}")
            print(f"Scene: {scene_data.uid}")
            print(f"Object: {bbox.label}, model_jid: {bbox.model_jid}")
            print("Missing preprocessing files.")
            raise SystemExit("Cannot proceed without real object features")

    scene_params['objfeat_vq_recon'] = np.array(objfeat_vq_recon)
    scene_params['objfeat_vq_indices'] = np.array(objfeat_vq_indices)
    scene_params['objfeat_vitg14_features'] = np.array(objfeat_vitg14_features)

    print(f"Successfully extracted features for {len(scene_data.bboxes)} objects")

    scene_params = validate_coordinates(scene_params, raw_dataset, verbose=not QUIET_MODE)

    return scene_params


def viz_org_scene(scene_data, raw_dataset, object_dataset, output_folder):
    """
    Visualize original scene without editing.
    """
    print("Visualizing original scene...")

    # Convert scene to required format - always use real features
    with tqdm(desc="Converting scene parameters with real features", unit="step") as pbar:
        scene_params = convert_scene_to_params(scene_data, raw_dataset)
        pbar.update(1)

    # Check original scene before 3D construction
    debug_check_scene_collisions(scene_data, "ORIGINAL_SCENE_VISUALIZATION", verbose=False)

    # Generate 3D scene - from room_edit.py construct_scene_from_vq_objdata
    try:
        with tqdm(desc="Constructing 3D scene", unit="step") as pbar:
            scene, scene_trimesh = construct_scene_from_vq_objdata(
                scene_params, object_dataset, all_classes=raw_dataset.class_labels
            )
            pbar.update(1)
    except Exception as e:
        print(f"Error constructing scene: {e}")
        return

    # Create output folders
    original_folder = os.path.join(output_folder, "original")
    os.makedirs(original_folder, exist_ok=True)

    # Export scene - from visualize.py export_scene()
    print("Exporting scene meshes...")
    with tqdm(desc="Exporting scene meshes", unit="step") as pbar:
        export_scene(original_folder, scene_trimesh)
        pbar.update(1)

    # Render with Blender - from room_edit.py get_blender_render()
    print("Rendering with Blender...")
    try:
        with tqdm(desc="Rendering scene", unit="step") as pbar:
            get_blender_render(scene_data, scene_trimesh,
                              save_folder=original_folder, verbose=False, remove_mesh=True, camera_dist=DEFAULT_CAMERA_DISTANCE)
            pbar.update(1)
        print(f"Original scene visualization saved to: {original_folder}")
    except Exception as e:
        print(f"Error rendering scene: {e}")


# create_output_folder removed - now imported from inference_utils


# construct_plan_prompt removed - now imported from inference_utils


# call_llm_api removed - now imported from inference_utils


# convert_single_plan removed - now imported from inference_utils


def process_command(scene_data, command, class_labels):
    """
    Process natural language command through LLM pipeline.
    """
    print("Processing command through LLM...")

    with tqdm(desc="Constructing LLM prompt", unit="step") as pbar:
        message = construct_plan_prompt(scene_data, command, class_labels, use_image=False)
        pbar.update(1)

    # Calling LLM API
    print("Calling OpenAI API...")
    with tqdm(desc="Calling OpenAI API", unit="request") as pbar:
        response = call_llm_api(message, OPENAI_API_KEY)
        if response is None:
            raise RuntimeError("Failed to get response from OpenAI API")
        pbar.update(1)


    print("=== RAW LLM RESPONSE ===")
    print(response)
    print("=== END RAW RESPONSE ===\n")

    print("Processing LLM response...")
    # Extract API commands from response - simplified regex search skipped LLaMA validation for now
    with tqdm(desc="Processing LLM response", unit="step") as pbar:
        processed_commands = extract_commands(response)
        pbar.update(1)

    print(f"Successfully processed command into {len(processed_commands)} instruction(s)")
    return processed_commands


# extract_commands removed - now imported from inference_utils


def prepare_single_scene_batch(scene_data, instructions, processed_dataset):
    """
    Convert single scene to batch format.
    Uses the processed dataset to get the properly encoded scene data.
    """

    print(f"Input instructions type: {type(instructions)}")
    print(f"Input instructions: {instructions}")

    # Creates proper instruction format for the model
    if isinstance(instructions, str):
        batch_instructions = [instructions]  # Single instruction
    elif isinstance(instructions, list):
        batch_instructions = [instructions]  # Wrap list for batch format
    else:
        batch_instructions = [str(instructions)]  # Fallback

    print(f"Batch instructions: {batch_instructions}")


    print(f"Creating basic batch structure for single scene editing")

    n_objects = len(scene_data.bboxes)
    translations = np.array([bbox.position for bbox in scene_data.bboxes])
    sizes = np.array([bbox.size for bbox in scene_data.bboxes])
    angles = np.array([[bbox.z_angle] for bbox in scene_data.bboxes])

    class_labels = [bbox.label for bbox in scene_data.bboxes]
    if hasattr(processed_dataset, 'class_labels'):
        class_label_to_idx = {label: idx for idx, label in enumerate(processed_dataset.class_labels)}
        class_indices = []
        for label in class_labels:
            if label in class_label_to_idx:
                class_indices.append(class_label_to_idx[label])
            else:

                print(f"Warning: Label '{label}' not found in processed dataset class labels")
                class_indices.append(0)
    else:
        class_indices = list(range(len(class_labels)))


    print("Extracting object features for batch preparation...")
    try:
        # Validate feature paths first
        validate_feature_paths(EDITROOM_DATA_FOLDER)

        # Extract REAL object features with NO fallbacks
        objfeat_vq_recon = []
        objfeat_vq_indices = []
        objfeat_vitg14_features = []

        for i, bbox in enumerate(scene_data.bboxes):
            print(f"Extracting features for object {i+1}/{n_objects}: {bbox.label}")
            features = extract_obj_features(bbox)
            objfeat_vq_recon.append(features['objfeat_vq_recon'])
            objfeat_vq_indices.append(features['objfeat_vq_indices'])
            objfeat_vitg14_features.append(features['objfeat_vitg14_features'])

        # Convert to numpy arrays
        objfeat_vq_recon = np.array(objfeat_vq_recon)
        objfeat_vq_indices = np.array(objfeat_vq_indices)
        objfeat_vitg14_features = np.array(objfeat_vitg14_features)

        print("Successfully extracted real object features for batch")

    except (FileNotFoundError, ValueError, SystemExit) as e:
        print(f"Real feature extraction failed: {e}")
        print("Missing preprocessing files.")
        raise SystemExit("Cannot proceed without real object features")

    # Normalize coordinates to [-1, 1] for model training
    if hasattr(processed_dataset, 'centroids') and hasattr(processed_dataset, 'sizes'):
        # Get normalization bounds from dataset
        centroids_min, centroids_max = processed_dataset.centroids
        sizes_min, sizes_max = processed_dataset.sizes

        translations_norm = 2.0 * (translations - centroids_min) / (centroids_max - centroids_min) - 1.0

        sizes_norm = 2.0 * (sizes - sizes_min) / (sizes_max - sizes_min) - 1.0

        print(f"Normalized translations: [{np.min(translations_norm):.3f}, {np.max(translations_norm):.3f}]")
        print(f"Normalized sizes: [{np.min(sizes_norm):.3f}, {np.max(sizes_norm):.3f}]")
    else:
        print("Warning: No normalization bounds available, using raw coordinates")
        translations_norm = translations
        sizes_norm = sizes

    angles_rad = angles * np.pi / 180.0
    angles_sincos = np.concatenate([
        np.sin(angles_rad),
        np.cos(angles_rad)
    ], axis=1)

    boxes = np.concatenate([
        translations_norm,
        sizes_norm,
        angles_sincos
    ], axis=1)

    # Create object masks (all objects are valid)
    obj_masks = np.ones(n_objects, dtype=np.float32)

    # Used the max_length from the dataset, 12 for bedroom 21 for living

    if hasattr(processed_dataset, 'max_length'):
        max_objects = processed_dataset.max_length
    else:
        room_type = None
        if hasattr(scene_data, 'scene_type'):
            room_type = str(scene_data.scene_type).lower()
        elif hasattr(scene_data, 'uid'):
            # Try to infer from scene ID
            uid_lower = scene_data.uid.lower()
            if 'bedroom' in uid_lower:
                room_type = 'bedroom'
            elif 'livingroom' in uid_lower:
                room_type = 'livingroom'
            elif 'diningroom' in uid_lower:
                room_type = 'diningroom'

        # Sets max_objects based on room type
        if room_type and 'bedroom' in room_type:
            max_objects = 12
        elif room_type and ('livingroom' in room_type or 'diningroom' in room_type):
            max_objects = 21
        else:
            max_objects = 12

    print(f"Using max_objects={max_objects} (from model training configuration)")


    num_edges = max_objects * (max_objects - 1) // 2
    edges_flat = np.zeros(num_edges, dtype=np.int64)

    edges_matrix = np.zeros((max_objects, max_objects), dtype=np.int64)

    # Pads arrays to max_objects size
    def pad_to_max(arr, max_size, pad_value=0):
        if len(arr) >= max_size:
            return arr[:max_size]
        padded = np.full((max_size,) + arr.shape[1:], pad_value, dtype=arr.dtype)
        padded[:len(arr)] = arr
        return padded

    objs_padded = pad_to_max(np.array(class_indices), max_objects, pad_value=processed_dataset.n_object_types)
    boxes_padded = pad_to_max(boxes, max_objects, pad_value=0)
    obj_masks_padded = pad_to_max(obj_masks, max_objects, pad_value=0)  # 0 for invalid objects
    objfeat_vq_recon_padded = pad_to_max(objfeat_vq_recon, max_objects, pad_value=0)

    objfeat_vq_indices_padded = pad_to_max(objfeat_vq_indices, max_objects, pad_value=64)
    objfeat_vitg14_features_padded = pad_to_max(objfeat_vitg14_features, max_objects, pad_value=0)


    sources = {
        'objs': torch.from_numpy(objs_padded).unsqueeze(0).long(),
        'boxes': torch.from_numpy(boxes_padded).unsqueeze(0).float(),
        'edges': torch.from_numpy(edges_matrix).unsqueeze(0).long(),
        'flatten_edges': torch.from_numpy(edges_flat).unsqueeze(0).long(),
        'obj_masks': torch.from_numpy(obj_masks_padded).unsqueeze(0).float(),
        'objfeat_vq_recon': torch.from_numpy(objfeat_vq_recon_padded).unsqueeze(0).float(),
        'objfeat_vq_indices': torch.from_numpy(objfeat_vq_indices_padded).unsqueeze(0).long(),
        'objfeat_vitg14_features': torch.from_numpy(objfeat_vitg14_features_padded).unsqueeze(0).float(),
        'room_uid': [scene_data.uid],
        'length': torch.tensor([n_objects]).long()
    }

    # Create batch structure
    batch = {
        'sources': sources,
        'targets': sources.copy(),  # Irrelevant for inference so I set it as a copy
        'instructions': batch_instructions  # Use properly formatted instructions
    }

    print(f"Final batch instructions: {batch['instructions']}")
    print(f"Batch created successfully")

    return batch


def params_to_scene_data(params, original_scene_data, object_dataset, all_classes):
    """
    Convert generated parameters back to scene_data format for next iteration.

    Args:
        params: Generated parameters from model (dict with translations, sizes, angles, etc.)
        original_scene_data: Original scene data for reference (can be None)
        object_dataset: Object dataset for finding furniture
        all_classes: Class labels

    Returns:
        updated scene_data object with updated objects
    """
    print("Converting parameters to scene data")


    from src.data.threed_front_scene import ThreedFutureModel
    from copy import deepcopy

    # Handle case where original_scene_data is None (for collision resolution)
    if original_scene_data is None:
        # Create a minimal scene data object
        class MinimalSceneData:
            def __init__(self):
                self.bboxes = []
                self.uid = "temp_scene"
        new_scene_data = MinimalSceneData()
    else:
        new_scene_data = deepcopy(original_scene_data)

    new_scene_data.bboxes = []

    if 'obj_masks' in params:
        valid_mask = params['obj_masks'] == 1
    else:

        valid_mask = np.ones(len(params['objs']), dtype=bool)

    num_valid_objects = np.sum(valid_mask)
    print(f"Valid objects: {num_valid_objects}")


    valid_indices = np.where(valid_mask)[0]

    for idx in valid_indices:

        obj_class_idx = params['objs'][idx]
        obj_class = all_classes[obj_class_idx]

        objfeat_vq_recon = params['objfeat_vq_recon'][idx]


        furniture_obj = object_dataset.get_closest_furniture_to_objfeat(obj_class, objfeat_vq_recon)

        if furniture_obj is None:
            print(f"Warning: Could not find furniture for class {obj_class}")
            continue


        position = params['translations'][idx]
        size = params['sizes'][idx]

        # Convert angle from sin/cos back to degrees if needed
        if params['angles'].shape[-1] == 2:
            angle_rad = np.arctan2(params['angles'][idx, 0], params['angles'][idx, 1])
            angle_deg = -angle_rad * 180.0 / np.pi
        else:
            angle_deg = params['angles'][idx, 0] * 180.0 / np.pi

        # Create new ThreedFutureModel object by copying from the closest furniture
        # This preserves all the mesh and feature information
        bbox = ThreedFutureModel(
            model_uid=furniture_obj.model_uid,
            model_jid=furniture_obj.model_jid,
            model_info=furniture_obj.model_info,
            position=list(position),  # Use generated position
            rotation=furniture_obj.rotation,  # Start with original rotation
            scale=furniture_obj.scale,  # Start with original scale
            path_to_models=furniture_obj.path_to_models
        )

        # Set the label (may be different from original if replaced)
        bbox.label = obj_class

        # Update the object's transformation parameters to match generated values
        bbox.position = list(position)  # Set position directly

        # Calculate scale factor to achieve the desired size
        original_size = furniture_obj.size
        if np.any(original_size > 0):
            size_scale_factor = size / original_size
            bbox.scale = [s * f for s, f in zip(furniture_obj.scale, size_scale_factor)]

        # Set rotation to achieve the desired z_angle
        # The rotation is [x, y, z, w] quaternion format, we want rotation around Y-axis
        angle_rad = angle_deg * np.pi / 180.0
        bbox.rotation = [0, np.sin(angle_rad/2), 0, np.cos(angle_rad/2)]

        print(f"Created {obj_class} at pos {position[:2]} with size {size[:2]}")

        new_scene_data.bboxes.append(bbox)

    print(f"Created {len(new_scene_data.bboxes)} objects in new scene")
    return new_scene_data


def generate_edited_scene(model, scene_data, processed_commands, processed_dataset,
                          output_folder=None, object_dataset=None, raw_dataset=None):
    """
    Generate edited scene using the model.

    For multiple commands, processes them sequentially where each command's
    output becomes the input for the next command.

    NEW: If output_folder, object_dataset, and raw_dataset are provided,
    intermediate scenes will be saved in edit_1/, edit_2/, ..., edit_N/ subdirectories.
    """
    print("Generating edited scene...")
    print(f"Number of commands: {len(processed_commands) if isinstance(processed_commands, list) else 1}")

    # Track if we should save intermediate results
    save_intermediate = (output_folder is not None and object_dataset is not None and raw_dataset is not None)
    if save_intermediate:
        print("INTERMEDIATE_SAVE: Intermediate scene saving enabled")

    # Check if we have multiple commands to process sequentially
    if isinstance(processed_commands, list) and len(processed_commands) > 1:
        print(f"Processing {len(processed_commands)} commands sequentially")

        # Start with the original scene
        current_scene_data = scene_data
        original_source_params = None  # Will store the original scene parameters
        all_intermediate_results = []

        # Process each command sequentially
        for i, command in enumerate(processed_commands):
            print(f"\nProcessing command {i+1}/{len(processed_commands)}: {command[:COMMAND_PREVIEW_LENGTH]}...")

            # Prepare batch with single command
            with tqdm(desc=f"Preparing batch for command {i+1}", unit="step") as pbar:
                batch = prepare_single_scene_batch(current_scene_data, command, processed_dataset)
                pbar.update(1)

            # Generate scene for this command
            with tqdm(desc=f"Generating scene for command {i+1}", unit="step") as pbar:
                results = model.predict_step(batch, 0)
                pbar.update(1)

            if not results or len(results) == 0:
                print(f"ERROR: No results generated for command {i+1}")
                break

            # Extract the generated parameters
            result_tuple = results[0]  # (source_params, target_params, generate_params, instruction)
            source_params, target_params, generate_params, instruction = result_tuple

            # Store the original source params from the first iteration
            if i == 0:
                original_source_params = source_params

            print(f"Command {i+1} completed")
            print(f"Generated objects: {np.sum(generate_params.get('obj_masks', 1) == 1) if 'obj_masks' in generate_params else 'unknown'}")

            # INTERMEDIATE_SAVE: Save intermediate result BEFORE any postprocessing
            if save_intermediate:
                print(f"INTERMEDIATE_SAVE: Saving intermediate result for command {i+1}")

                # Debug: Check parameters before copying
                print(f"DEBUG: Before copy - generate_params keys: {list(generate_params.keys())}")
                if 'translations' in generate_params:
                    print(f"DEBUG: Before copy - translations range: [{np.min(generate_params['translations']):.3f}, {np.max(generate_params['translations']):.3f}]")

                # Create deep copy to preserve normalized parameters for saving
                result_tuple_copy = deepcopy(result_tuple)
                print(f"DEBUG: Deep copy created successfully")

                # Create subdirectory for this edit step
                edit_folder = os.path.join(output_folder, f"edit_{i+1}")
                print(f"INTERMEDIATE_SAVE: Creating directory {edit_folder}")

                # Save using exact same logic as final output
                try:
                    with tqdm(desc=f"Saving intermediate scene {i+1}", unit="step") as pbar:
                        save_results(result_tuple_copy, edit_folder, object_dataset, raw_dataset, model)
                        pbar.update(1)
                    print(f"INTERMEDIATE_SAVE: Successfully saved edit_{i+1}")
                except Exception as e:
                    print(f"INTERMEDIATE_SAVE: Error saving edit_{i+1}: {e}")
                    import traceback
                    traceback.print_exc()

                # Clean up copy from memory to avoid interference
                del result_tuple_copy
                print(f"DEBUG: Deep copy cleaned up")

            # Store intermediate result (for potential future use)
            all_intermediate_results.append(result_tuple)

            # For the last command, we're done
            if i == len(processed_commands) - 1:
                print("Final command processed")
                # Return the final result with the original source and full instruction list
                final_result = (original_source_params, target_params, generate_params, processed_commands)

                if save_intermediate:
                    print("INTERMEDIATE_SAVE: All intermediate scenes saved. Final scene will be saved by main().")

                return [final_result]

            # Convert generated parameters back to scene_data for next iteration
            print(f"Converting output to scene_data for next command")

            # Debug: Check parameters before denormalization
            print(f"DEBUG: Before denormalization - generate_params id: {id(generate_params)}")
            if 'translations' in generate_params:
                print(f"DEBUG: Before denormalization - translations range: [{np.min(generate_params['translations']):.3f}, {np.max(generate_params['translations']):.3f}]")

            # First denormalize the parameters (this modifies generate_params in-place)
            generate_params_denorm = model.postprocess_func(generate_params)

            # Debug: Check parameters after denormalization
            print(f"DEBUG: After denormalization - generate_params id: {id(generate_params)} (same object)")
            print(f"DEBUG: After denormalization - generate_params_denorm id: {id(generate_params_denorm)}")
            if 'translations' in generate_params:
                print(f"DEBUG: After denormalization - generate_params translations range: [{np.min(generate_params['translations']):.3f}, {np.max(generate_params['translations']):.3f}]")
            if 'translations' in generate_params_denorm:
                print(f"DEBUG: After denormalization - generate_params_denorm translations range: [{np.min(generate_params_denorm['translations']):.3f}, {np.max(generate_params_denorm['translations']):.3f}]")
            print(f"DEBUG: Are they the same object? {generate_params is generate_params_denorm}")

            # Apply collision resolution if available
            if COLLISION_DETECTION_AVAILABLE:
                generate_params_denorm = apply_collision_resolution_to_params(
                    generate_params_denorm, model.object_dataset, model.all_classes
                )

            # Then convert to scene_data
            current_scene_data = params_to_scene_data(
                generate_params_denorm,
                current_scene_data,
                model.object_dataset,
                model.all_classes
            )

            print(f"New scene has {len(current_scene_data.bboxes)} objects")

            # Check for collisions after each command
            debug_check_scene_collisions(current_scene_data, f"After command {i+1}")

    else:
        # Single command or already formatted for batch processing
        print("Processing single command")

        # Convert single scene to batch format
        with tqdm(desc="Preparing scene batch", unit="step") as pbar:
            batch = prepare_single_scene_batch(scene_data, processed_commands, processed_dataset)
            pbar.update(1)

        # Use model generation
        with tqdm(desc="Generating scene", unit="step") as pbar:
            results = model.predict_step(batch, 0)
            pbar.update(1)

        return results  # Return full results tuple


def save_results(data_tuple, output_folder, object_dataset, raw_dataset, model):
    """
    Args:
        data_tuple: Tuple of (source_params, target_params, generate_params, instructions)
        output_folder: Where to save results
        object_dataset: Object dataset for scene construction
        raw_dataset: Raw dataset for scene rendering
        model: Model for postprocessing
    """
    print("Saving inference results...")

    source_params, target_params, generate_params, instructions = data_tuple

    os.makedirs(output_folder, exist_ok=True)

    # Saves all_data.pt in same format (target as None)
    with tqdm(desc="Saving data file", unit="step") as pbar:
        inference_data_tuple = (source_params, None, generate_params, instructions)  # None for target
        torch.save(inference_data_tuple, os.path.join(output_folder, "all_data.pt"))
        pbar.update(1)

        print("Postprocessing generated scene (denormalizing coordinates)...")
    with tqdm(desc="Postprocessing generated scene", unit="step") as pbar:
        generate_params_denorm = model.postprocess_func(generate_params)
        pbar.update(1)


    print("COLLISION_DEBUG:")
    print(f"  Number of objects: {len(generate_params_denorm.get('objs', []))}")
    if 'translations' in generate_params_denorm:
        trans = generate_params_denorm['translations']
        print(f"  Translation range: [{np.min(trans):.3f}, {np.max(trans):.3f}]")
    if 'sizes' in generate_params_denorm:
        sizes = generate_params_denorm['sizes']
        print(f"  Size range: [{np.min(sizes):.3f}, {np.max(sizes):.3f}]")
    if 'obj_masks' in generate_params_denorm:
        valid_objects = np.sum(generate_params_denorm['obj_masks'] == 1)
        print(f"  Valid objects (from mask): {valid_objects}")

    # Apply collision resolution before constructing 3D scene
    if COLLISION_DETECTION_AVAILABLE:
        generate_params_denorm = apply_collision_resolution_to_params(
            generate_params_denorm, object_dataset, model.all_classes
        )

    print("Constructing generated 3D scene...")
    with tqdm(desc="Constructing generated scene", unit="step") as pbar:
        generate_scene, generate_scene_trimesh = construct_scene_from_vq_objdata(
            generate_params_denorm, object_dataset, all_classes=model.all_classes)
        pbar.update(1)

    # Check final generated scene for collisions
    try:
        # Create a temporary scene_data object for collision checking
        from src.data.threed_front_scene import ThreedFutureModel
        class TempScene:
            def __init__(self):
                self.bboxes = []

        temp_scene = TempScene()


        for scene_obj in generate_scene:
            bbox = ThreedFutureModel(
                model_uid=scene_obj['obj'].model_uid,
                model_jid=scene_obj['obj'].model_jid,
                model_info=scene_obj['obj'].model_info,
                position=list(scene_obj['translation']),
                rotation=[0, 0, 0, 1],
                scale=[1, 1, 1],
                path_to_models=scene_obj['obj'].path_to_models
            )
            bbox.label = scene_obj['obj'].label
            bbox.size = scene_obj['size']

            R = scene_obj['rotation']
            z_angle = np.arctan2(R[2, 0], R[0, 0])
            bbox.z_angle = z_angle
            temp_scene.bboxes.append(bbox)

        debug_check_scene_collisions(temp_scene, "Final generated scene")

    except Exception as e:
        pass  # Silently skip if collision check fails

    generate_image_folder = os.path.join(output_folder, "generate")

    print("Rendering generated scene...")
    try:
        with suppress_output():
            try:
                with tqdm(desc="Rendering generated scene (PyRender)", unit="step", disable=True) as pbar:
                    pred, pred_images, scene_centroid, radius = render_generated_scene(
                        generate_scene_trimesh, output_folder=generate_image_folder)
                    pbar.update(1)
                print("Successfully rendered with PyRender")
            except Exception as e:
                raise e  # Re-raise to trigger Blender fallback
    except Exception as e:

        print("PyRender unavailable, using Blender instead...")

        try:
            actual_scene_uid = source_params['room_uid']
            if actual_scene_uid in raw_dataset.uid_to_scene_index:
                source_index = raw_dataset.uid_to_scene_index[actual_scene_uid]
                raw_source_scene = raw_dataset[source_index]

                with tqdm(desc="Rendering generated scene (Blender)", unit="step") as pbar:
                    get_blender_render(raw_source_scene, generate_scene_trimesh,
                                      save_folder=generate_image_folder, verbose=False, remove_mesh=True, camera_dist=DEFAULT_CAMERA_DISTANCE)
                    pbar.update(1)
            else:
                print(f"Warning: Scene UID '{actual_scene_uid}' not found in dataset index")
                print("Available UIDs:", list(raw_dataset.uid_to_scene_index.keys())[:MAX_FILE_MATCHES_DISPLAY], "...")
        except Exception as e2:
            print(f"Error in Blender rendering: {e2}")

    with tqdm(desc="Saving instruction file", unit="step") as pbar:
        instruction_save_path = os.path.join(output_folder, "instruction.json")
        with open(instruction_save_path, "w") as f:
            json.dump(instructions, f, indent=2)
        pbar.update(1)

    with tqdm(desc="Saving info file", unit="step") as pbar:
        info_save_path = os.path.join(output_folder, "inference_info.json")
        info_data = {
            "mode": "inference_only",
            "scene_uid": source_params['room_uid'],
            "num_objects": len(generate_params_denorm['objs']),
            "instruction": instructions,
            "coordinate_denormalization": "applied",
            "note": "No target comparison available in inference mode"
        }
        with open(info_save_path, "w") as f:
            json.dump(info_data, f, indent=2)
        pbar.update(1)

    print(f"Inference results saved to: {output_folder}")
    print(f"Generated images: {generate_image_folder}")
    print("Target comparison skipped in inference mode")

    return {"mode": "inference_only"}


def resolve_collisions(scene_data, max_attempts=MAX_COLLISION_RESOLUTION_ATTEMPTS):
    """
    Resolve collisions in a scene by adjusting object positions.

    """
    if not COLLISION_DETECTION_AVAILABLE:
        return scene_data


    initial_collisions = debug_check_scene_collisions(scene_data, "BEFORE_RESOLUTION", verbose=True)

    if not initial_collisions:
        return scene_data


    from copy import deepcopy
    resolved_scene = deepcopy(scene_data)
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        collision_found = False


        for i in range(len(resolved_scene.bboxes)):
            for j in range(i + 1, len(resolved_scene.bboxes)):
                if two_rectangle_collision(resolved_scene.bboxes[i], resolved_scene.bboxes[j]):
                    collision_found = True
                    obj_i = resolved_scene.bboxes[i]
                    obj_j = resolved_scene.bboxes[j]


                    pos_i = np.array(obj_i.position)
                    pos_j = np.array(obj_j.position)


                    direction = pos_i - pos_j
                    direction[1] = 0

                    if np.linalg.norm(direction[[0, 2]]) < COLLISION_SAFETY_MARGIN / 10:

                        angle = np.random.uniform(0, 2 * np.pi)
                        direction = np.array([np.cos(angle), 0, np.sin(angle)])
                    else:
                        direction = direction / np.linalg.norm(direction)



                    size_i = np.array(obj_i.size)
                    size_j = np.array(obj_j.size)


                    separation_needed = (size_i[0] + size_j[0]) * abs(direction[0]) + \
                                      (size_i[2] + size_j[2]) * abs(direction[2]) + COLLISION_SAFETY_MARGIN


                    if j == len(resolved_scene.bboxes) - 1:

                        move_distance = separation_needed * COLLISION_SEPARATION_FACTOR
                        new_pos = list(pos_j + direction * move_distance)
                        new_pos[1] = obj_j.position[1]
                        obj_j.position = new_pos
                    else:

                        move_distance = separation_needed * COLLISION_SEPARATION_FACTOR
                        new_pos = list(pos_i - direction * move_distance)
                        new_pos[1] = obj_i.position[1]  #
                        obj_i.position = new_pos

                    break

            if collision_found:
                break

        if not collision_found:
            print(f"RESOLVED: Collisions resolved in {attempts} iteration(s)")
            break

    if attempts >= max_attempts:
        print(f"WARNING: Some collisions remain after {max_attempts} attempts")


    debug_check_scene_collisions(resolved_scene, "AFTER_RESOLUTION", verbose=True)

    return resolved_scene


def apply_collision_resolution_to_params(generate_params, object_dataset, all_classes):
    """
    Apply collision resolution to generated parameters before scene construction.
    This creates a temporary scene, resolves collisions, then extracts parameters back.
    """


    temp_scene_data = params_to_scene_data(
        generate_params,
        None,
        object_dataset,
        all_classes
    )


    resolved_scene_data = resolve_collisions(temp_scene_data)


    resolved_params = generate_params.copy()


    if len(resolved_scene_data.bboxes) > 0:
        new_translations = []
        new_sizes = []
        new_angles = []

        for bbox in resolved_scene_data.bboxes:
            new_translations.append(bbox.position)
            new_sizes.append(bbox.size)


            if hasattr(bbox, 'z_angle'):
                angle = bbox.z_angle
            else:

                angle = DEFAULT_ZERO_ANGLE

            angle_rad = angle * np.pi / 180.0 if abs(angle) > 2*np.pi else angle
            new_angles.append([np.sin(angle_rad), np.cos(angle_rad)])

        resolved_params['translations'] = np.array(new_translations)
        resolved_params['sizes'] = np.array(new_sizes)
        resolved_params['angles'] = np.array(new_angles)

    return resolved_params


def main():
    """Main execution function."""
    args = parse_arguments()

    # Set quiet mode globally
    global QUIET_MODE
    QUIET_MODE = args.quiet


    if not OPENAI_API_KEY:
        if not args.no_edit:
            print("ERROR: OPENAI_API_KEY not set")
            return 1
        else:
            print("Warning: OPENAI_API_KEY not set, but --no_edit mode doesn't require it.")

    if not os.path.exists(BLENDER_PATH):
        print(f"Warning: Blender not found at {BLENDER_PATH}")

    # Set random seed
    seed_everything(args.seed)

    try:
        # Load configurations
        with tqdm(desc="Loading configurations", unit="file") as pbar:
            sg_config, sg2sc_config = load_configs(args.sg_config_file, args.sg2sc_config_file, verbose=not QUIET_MODE)
            pbar.update(1)

        scene_data, raw_dataset, object_dataset = load_single_scene(args.source_scene_id, args.room_type)


        debug_check_scene_collisions(scene_data, "Original scene")

        # Creates temp dataset for getting properties
        print("Preparing dataset for model...")
        with tqdm(desc="Preparing dataset properties", unit="step") as pbar:

            class_labels = raw_dataset.class_labels
            n_object_types = len(raw_dataset.object_types)
            n_predicate_types = len(getattr(raw_dataset, 'predicate_types', ['left', 'right', 'front', 'behind']))
            pbar.update(1)


        class DatasetWrapper:
            def __init__(self, n_obj_types, n_pred_types, class_labels, max_length, raw_dataset):
                self.n_object_types = n_obj_types
                self.n_predicate_types = n_pred_types
                self.class_labels = class_labels
                self.max_length = max_length
                self.raw_dataset = raw_dataset

                # bounds copied from raw_dataset for normalization
                if hasattr(raw_dataset, 'centroids'):
                    self.centroids = raw_dataset.centroids
                if hasattr(raw_dataset, 'sizes'):
                    self.sizes = raw_dataset.sizes
                if hasattr(raw_dataset, 'angles'):
                    self.angles = raw_dataset.angles

                self.post_process = self._create_denormalization_function()

            def _create_denormalization_function(self):
                """Create a denormalization function that converts model output back to real-world scale."""
                def denormalize_scene_params(sample_params):
                    """
                    Denormalize coordinates from [-1,1] back to real-world scale.
                    This is essential because the model outputs normalized coordinates.
                    """
                    print("Denormalizing scene parameters...")

                    if hasattr(self, 'centroids') and hasattr(self, 'sizes'):
                        centroids_min, centroids_max = self.centroids
                        sizes_min, sizes_max = self.sizes


                        if 'translations' in sample_params:
                            normalized_vals = sample_params['translations']
                            denormalized = Scale.descale(normalized_vals, centroids_min, centroids_max)
                            sample_params['translations'] = denormalized
                            print(f"Denormalized translations: [{np.min(denormalized):.3f}, {np.max(denormalized):.3f}]")


                        if 'sizes' in sample_params:
                            normalized_vals = sample_params['sizes']
                            denormalized = Scale.descale(normalized_vals, sizes_min, sizes_max)
                            sample_params['sizes'] = denormalized
                            print(f"Denormalized sizes: [{np.min(denormalized):.3f}, {np.max(denormalized):.3f}]")


                        if 'angles' in sample_params:
                            angles_sincos = sample_params['angles']
                            if angles_sincos.shape[-1] == 2:
                                angles_rad = np.arctan2(angles_sincos[..., 1:2], angles_sincos[..., 0:1])
                                sample_params['angles'] = angles_rad
                                print(f"Converted angles from sin/cos to radians")
                    else:
                        print("No bounds available for denormalization")

                    return sample_params

                return denormalize_scene_params

        from src.data.threed_front_dataset_base import Scale

        # Gets max_length from the sg_config
        max_length = sg_config['data']['max_length']
        print(f"Using max_length={max_length} from sg_config")

        dataset = DatasetWrapper(n_object_types, n_predicate_types, class_labels, max_length, raw_dataset)


        with tqdm(desc="Creating output folder", unit="step") as pbar:
            output_folder = create_output_folder(args.output_directory, args.source_scene_id)
            print(f"Output folder: {output_folder}")
            pbar.update(1)

        if args.no_edit:

            viz_org_scene(scene_data, raw_dataset, object_dataset, output_folder)
            print("Original scene visualization completed.")
            return 0

        print("\n=== Starting User Interactive Editing Pipeline ===")


        # Gets natural language command from user
        command = input("\nEnter your editing command: ")
        if not command.strip():
            print("No command provided. Exiting.")
            return 1

        print(f"Processing command: '{command}'")

        # Process command through LLM pipeline
        with tqdm(desc="Preparing class labels", unit="step") as pbar:
            class_labels_list = [c.replace("_", " ") for c in class_labels[:-2]]
            pbar.update(1)

        try:
            processed_commands = process_command(scene_data, command, class_labels_list)
            print(f"Processed commands: {processed_commands}")
        except Exception as e:
            print(f"Error processing natural language command: {e}")
            print("Please try a different command or check your OpenAI API key.")
            return 1

        print("Loading models directly for single scene editing...")
        try:

            sg_config["load_path"] = args.sg_weight_file
            sg2sc_config["load_path"] = args.sg2sc_weight_file


            checkpoint = torch.load(args.sg_weight_file, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                checkpoint_num_objs = None
                checkpoint_num_preds = None

                for key in state_dict.keys():
                    if 'model.network.node_embed.0.weight' in key:
                        node_dim = state_dict[key].shape[0]
                        checkpoint_num_objs = node_dim - 2  # -2 for empty node and [mask] token
                        print(f"Detected node_dim={node_dim}, so num_objs={checkpoint_num_objs}")
                    elif 'model.network.edge_embed.0.weight' in key:
                        edge_dim = state_dict[key].shape[0]
                        checkpoint_num_preds = edge_dim - 2  # -2 for empty edge and [mask] token
                        print(f"Detected edge_dim={edge_dim}, so num_preds={checkpoint_num_preds}")

                if checkpoint_num_objs is None:
                    checkpoint_num_objs = dataset.n_object_types
                    print(f"Could not detect object types from checkpoint, using dataset default: {checkpoint_num_objs}")

                if checkpoint_num_preds is None:
                    checkpoint_num_preds = dataset.n_predicate_types
                    print(f"Could not detect predicate types from checkpoint, using dataset default: {checkpoint_num_preds}")
            else:
                checkpoint_num_objs = dataset.n_object_types
                checkpoint_num_preds = dataset.n_predicate_types


            if checkpoint_num_objs <= len(raw_dataset.object_types):
                truncated_object_types = raw_dataset.object_types[:checkpoint_num_objs]
                truncated_class_labels = truncated_object_types + ["start", "end"]
            else:
                truncated_class_labels = dataset.class_labels
            print(f"Using truncated class labels: {len(truncated_class_labels)} classes (object types: {checkpoint_num_objs})")

            model = RoomEdit(
                sg_config, sg2sc_config,
                num_objs=checkpoint_num_objs,  # Use checkpoint number
                num_preds=checkpoint_num_preds,
                pred_save_folder=None,
                postprocess_func=dataset.post_process,
                object_dataset=object_dataset,
                all_classes=truncated_class_labels,  # Use truncated list
                raw_dataset=raw_dataset
            )
            model.eval()
            print("Direct model loading successful!")

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please check that the model checkpoint files exist and are compatible.")
            return 1

        # Generate edited scene
        try:
            print(f"\nStarting scene generation with {len(processed_commands)} commands")
            with tqdm(desc="Generating edited scene", unit="step") as pbar:
                results = generate_edited_scene(model, scene_data, processed_commands, dataset,
                                              output_folder, object_dataset, raw_dataset)
                pbar.update(1)
            print(f"Scene generation completed, got {len(results) if results else 0} results")
        except Exception as e:
            print(f"Error during scene generation: {e}")
            import traceback
            traceback.print_exc()
            return 1


        if not results or len(results) == 0:
            print("Error: No results generated")
            return 1

        data_tuple = results[0]  # [source_i, target_i, generate_i, instruction_i]
        source_data = data_tuple[0]
        target_data = data_tuple[1]
        generated_data = data_tuple[2]
        processed_instruction = data_tuple[3]

        print("Results extracted successfully")

        # Determine if we processed multiple commands (intermediate scenes already saved)
        is_multi_step = isinstance(processed_commands, list) and len(processed_commands) > 1

        if is_multi_step:
            # For multi-step, save final result in edit_N/ format to match intermediate structure
            final_edit_num = len(processed_commands)
            final_edit_folder = os.path.join(output_folder, f"edit_{final_edit_num}")
            print(f"Multi-step editing: Saving final result as edit_{final_edit_num}")

            try:
                save_results(data_tuple, final_edit_folder, object_dataset, raw_dataset, model)
            except Exception as e:
                print(f"Error saving final results: {e}")
                import traceback
                traceback.print_exc()
                return 1

            print(f"Multi-step editing completed successfully")
            print(f"Intermediate results: {output_folder}/edit_1/ through {output_folder}/edit_{final_edit_num-1}/")
            print(f"Final result: {output_folder}/edit_{final_edit_num}/")
        else:
            # For single-step, use original saving logic
            print("Single-step editing: Using original save location")
            try:
                save_results(data_tuple, output_folder, object_dataset, raw_dataset, model)
            except Exception as e:
                print(f"Error saving results: {e}")
                import traceback
                traceback.print_exc()
                return 1

            print(f"Single-step editing completed successfully")
            print(f"Results saved to: {output_folder}")
            print(f"Generated images available in: {os.path.join(output_folder, 'generate')}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
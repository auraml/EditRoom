from huggingface_hub import snapshot_download
import os
import shutil

# Create weights directory
os.makedirs("weights", exist_ok=True)

print("Downloading EditRoomOrg weights...")
snapshot_download(
    repo_id="auraml/auragen",
    repo_type="model",
    allow_patterns="EditRoomOrg/weights/*",
    local_dir="temp_download",
    local_dir_use_symlinks=False
)

# Move files from nested structure to weights folder
source_path = "temp_download/EditRoomOrg/weights"
if os.path.exists(source_path):
    # Copy all files from source to weights directory
    for item in os.listdir(source_path):
        source_file = os.path.join(source_path, item)
        dest_file = os.path.join("weights", item)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, dest_file)
        elif os.path.isdir(source_file):
            shutil.copytree(source_file, dest_file, dirs_exist_ok=True)

    # Clean up temporary directory
    shutil.rmtree("temp_download")
    print("EditRoomOrg weights downloaded into ./weights/")
else:
    print("Warning: EditRoomOrg/weights folder not found in the repository")
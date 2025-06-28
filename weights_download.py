from huggingface_hub import snapshot_download
import os
os.makedirs("weights", exist_ok=True)

print("Downloading weights from EditRoomOrg repository...")

snapshot_download(

    repo_id="auraml/EditRoomOrg",

    repo_type="model",

    allow_patterns="weights/*",

    local_dir="./",

    local_dir_use_symlinks=False

)

print("EditRoomOrg weights downloaded into ./weights/")
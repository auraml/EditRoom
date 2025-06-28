import os
from dotenv import load_dotenv
load_dotenv()

EDITROOM_DATA_FOLDER = os.getenv("EDITROOM_DATA_FOLDER", "./datasets")
print(
    f"EDITROOM_DATA_FOLDER is set to {EDITROOM_DATA_FOLDER}. "
    "If you want to change the path, set the EDITROOM_DATA_FOLDER environment variable."
)
PATH_TO_SCENE = os.path.join(EDITROOM_DATA_FOLDER, "3D-FRONT/3D-FRONT")
print(PATH_TO_SCENE)
PATH_TO_MODEL = os.path.join(EDITROOM_DATA_FOLDER, "3D-FRONT/3D-FUTURE-model")
print(PATH_TO_MODEL)
EDIT_DATA_FOLDER = os.path.join(EDITROOM_DATA_FOLDER, "editroom_dataset")
print(
    f"EDIT_DATA_FOLDER is set to {EDIT_DATA_FOLDER}. "
    "If you want to change the path, set the EDITROOM_DATA_FOLDER environment variable."
)
PATH_TO_PREPROCESS = os.path.join(EDITROOM_DATA_FOLDER, "preprocess")
print(
    f"PATH_TO_PREPROCESS is set to {PATH_TO_PREPROCESS}. "
    "If you want to change the path, set the EDITROOM_DATA_FOLDER environment variable."
)

# Place API Keys and Blender path here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
BLENDER_PATH = os.getenv("BLENDER_PATH", "./blender/blender-3.3.1-linux-x64/blender")

print(
    f"BLENDER_PATH is set to {BLENDER_PATH}. "
    "Make sure Blender is installed at this location."
)


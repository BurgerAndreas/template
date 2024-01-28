# Real programmers would use environment variables but I am a physicist
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOWNLOAD_DIR = f"{ROOT_DIR}/data"
OUTPUT_DIR = f"{ROOT_DIR}/output"
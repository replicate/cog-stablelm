#!/usr/bin/env python

import os
import shutil
from distutils.dir_util import copy_tree
from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL_REPO = "stabilityai/stablelm-tuned-alpha-7b"
LOCAL_MODEL_DIR = "./pretrained_weights"
LOCAL_TOKENIZER_DIR = "./tokenizer"

if os.path.exists(LOCAL_MODEL_DIR):
    shutil.rmtree(LOCAL_MODEL_DIR)

# setup temporary directory
with TemporaryDirectory() as tmpdir:
    # download snapshot
    snapshot_dir = snapshot_download(
        repo_id=MODEL_REPO,
        cache_dir=tmpdir,
        allow_patterns=["*.bin", "*.json", "*.md"],
    )
    # copy snapshot to model dir
    copy_tree(snapshot_dir, str(LOCAL_MODEL_DIR))

# Move tokenizer to separate location
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, padding_side="left")
tokenizer.save_pretrained(LOCAL_TOKENIZER_DIR)

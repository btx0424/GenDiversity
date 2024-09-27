import torch
import numpy as np
import os.path as osp
import json
import matplotlib.pyplot as plt
import yaml

from collections import OrderedDict

FILE_PATH = osp.dirname(__file__)

model_name = "all-MiniLM-L6-v2"
model_name = "all-mpnet-base-v2"
cache_folder = ".cache"

try:
    from cliport.tasks import names, new_names
    template_tasks = {
        name: f"Task: {name} \n Description: {cls.__doc__}" 
        for name, cls in names.items()
    }
    generated_tasks = {
        name: f"Task: {name} \n Description: {cls.__doc__}" 
        for name, cls in new_names.items()
    }
except Exception as e:
    raise e

piles = []
stack = []
place = []
build = []
for key, val in generated_tasks.items():
    if "pile" in key:
        piles.append(val)
    if "stack" in key:
        stack.append(val)
    if "place" in key:
        place.append(val)
    if "build" in key:
        build.append(val)
for key, val in template_tasks.items():
    if "pile" in key:
        piles.append(val)
    if "stack" in key:
        stack.append(val)
    if "place" in key:
        place.append(val)
    if "build" in key:
        build.append(val)
descs = {
    "pile": piles,
    "stack": stack,
    "place": place,
    "build": build
}
yaml.safe_dump(descs, open("gensim_descs.yaml", "w"))
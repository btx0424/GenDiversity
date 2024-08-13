import torch
import numpy as np
import os.path as osp
import json
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster._kmeans import KMeans
from collections import OrderedDict

FILE_PATH = osp.dirname(__file__)

model_name = "all-MiniLM-L6-v2"
model_name = "all-mpnet-base-v2"
cache_folder = ".cache"

model = SentenceTransformer(model_name, cache_folder=cache_folder)

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

print(template_tasks)

# task_names = np.array(list(desc.keys()))
# desc_emb = model.encode(list(desc.values()))

# n_clusters = 8
# k_means = KMeans(n_clusters=n_clusters)
# k_means.fit(desc_emb)
# labels = k_means.labels_

# groups = {}
# for i in range(n_clusters):
#     groups[i] = task_names[labels == i]

# print(groups)
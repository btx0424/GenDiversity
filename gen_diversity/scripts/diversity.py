import yaml
import torch
import numpy as np
import os.path as osp
import json
import matplotlib.pyplot as plt
import openai
import requests
import os
import argparse

from sentence_transformers import SentenceTransformer, util
from collections import OrderedDict

FILE_PATH = osp.dirname(__file__)


def sentence_tf(descs, model_name = "all-mpnet-base-v2"):
    cache_folder = ".cache"

    model = SentenceTransformer(model_name, cache_folder=cache_folder)

    emb = model.encode(descs)
    sim = util.cos_sim(emb, emb)
    # remove self-similarity
    eye = np.eye(sim.shape[0], dtype=bool)
    sim = sim[~eye].reshape(sim.shape[0],sim.shape[0]-1)
    diversity = - sim.mean(1).log().mean().item()
    print(diversity)
    return diversity



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default="gensim")
    parser.add_argument('-g', '--group', type=str, default="all")
    args = parser.parse_args()

    project = args.project
    group = args.group

    if project == "bbsea":
        descs = yaml.safe_load(open(f"texts/bbsea/{group}.yaml"))
        descs = list(descs)
    elif project == "robogen":
        descs = yaml.safe_load(open(f"texts/robogen/{group}.yaml"))
        descs = list(descs.values())
    elif project == "gensim":
        descs = yaml.safe_load(open(f"texts/gensim_descs.yaml"))
        if group == "all": # combine all lists
            descs = sum(descs.values(), [])
        else:
            descs = list(descs[group])
    else:
        raise ValueError(f"Unknown project {project}")
    
    print("Total number of descriptions:", len(descs))
    diversity = sentence_tf(descs)


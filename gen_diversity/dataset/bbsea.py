import os
import torch
import numpy as np
import einops
import inspect
import time
import random

from tqdm import tqdm
from torch.utils.data import Dataset
from tensordict import TensorDict, MemoryMappedTensor
from torchvision.transforms.functional import resize
from collections import defaultdict
from typing import Mapping


class BBSEADataset(Dataset):
    def __init__(self, data: TensorDict, seq_length: int):
        super().__init__()
        self.data = data
        self.total_length = data.shape[0]
        self.seq_length = seq_length

        self.starts = self.data["is_first"].nonzero().squeeze(-1)
        self.ends = self.data["is_terminal"].nonzero().squeeze(-1)
    
    def __len__(self):
        return self.total_length - self.seq_length + 1
    
    def __getitem__(self, index):
        if isinstance(index, int):
            data = self.data[index: index+self.seq_length]
        else:
            data = torch.stack([self[i] for i in index])
        return data
    
    @classmethod
    def make(cls, root_dir: str, seq_length: int, max_episodes: int = None):
        
        episode_data_paths = []
        num_tasks = 0
        for task in tqdm(os.listdir(root_dir)):
            task_path = os.path.join(root_dir, task)
            episode_paths = [
                os.path.join(task_path, path)
                for path in os.listdir(task_path) 
                if path.endswith(".pt")
            ]
            print(f"{task}: {len(episode_paths)} episodes")
            episode_data_paths.extend(episode_paths)
            num_tasks += 1
            

        random.seed(0)
        random.shuffle(episode_data_paths)
        print(num_tasks, len(episode_data_paths))
        for i in range(5):
            print(episode_data_paths[i])
        data = []
        for episode_path in episode_data_paths:
            episode_data = torch.load(episode_path)
            data.append(TensorDict(episode_data, [len(episode_data["is_first"])]))

        data = torch.cat(data, dim=0)
        print(data)
        return cls(data, seq_length)

def group_tasks(root_path: str):
    drawer_dir = os.path.join(root_path, "drawer")
    table_dir = os.path.join(root_path, "table")
    trajectory_path = os.path.join(root_path, "trajectory")
    
    os.makedirs(drawer_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    num_tasks = 0
    for scene_id in sorted(os.listdir(trajectory_path), key=lambda x: int(x)):
        scene_path = os.path.join(trajectory_path, scene_id)
        for task_desc in os.listdir(scene_path):
            print(f"Scene: {scene_id}, Task: {task_desc}")
            if "drawer" in task_desc and not os.path.exists(os.path.join(drawer_dir, f"{scene_id}_{task_desc}")):
                os.symlink(os.path.join(scene_path, task_desc), os.path.join(drawer_dir, f"{scene_id}_{task_desc}"))
            elif not os.path.exists(os.path.join(table_dir, f"{scene_id}_{task_desc}")):
                os.symlink(os.path.join(scene_path, task_desc), os.path.join(table_dir, f"{scene_id}_{task_desc}"))
            num_tasks += 1
    print(f"Total tasks: {num_tasks}")


if __name__ == "__main__":
    root_path = "/localdata/bxu/isaac_lab/bbsea/your_path_to_output"
    group_tasks(root_path)
    BBSEADataset.make("/localdata/bxu/isaac_lab/bbsea/your_path_to_output/table", 40)
    
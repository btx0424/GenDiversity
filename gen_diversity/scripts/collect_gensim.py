"""Data collection script."""

import os
import hydra
import numpy as np
import torch
import cliport
import termcolor

from cliport import tasks
import gen_diversity
from gen_diversity.dataset import GENSIM_ROOT, GenSimEnvironment

import random
import logging
import time
import pybullet as p
import imageio
import einops
from collections import defaultdict


@hydra.main(config_path=f'{GENSIM_ROOT}/cliport/cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.

    env = GenSimEnvironment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=240,
        record_cfg=cfg['record']
    )
    cfg['task'] = cfg['task'].replace("_", "-")
    task: tasks.Task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']

    agent = task.oracle(env)
    dataset_path = os.path.join(GENSIM_ROOT, "data", cfg["mode"], cfg["task"])
    print(f"Dataset Path: {dataset_path}")
    
    os.makedirs(dataset_path, exist_ok=True)
    msg = (
        f"Collecting data for task {cfg['task']}, mode {cfg['mode']}"
        f"into {dataset_path}."
    )
    logging.info(termcolor.colored(msg, "green"))
    
    # meta_path = os.path.join(dataset_path, "meta.pt")
    # if not os.path.exists(meta_path):
    #     from sentence_transformers import SentenceTransformer
    #     model = SentenceTransformer("all-MiniLM-L6-v2")
    #     task_desc = f"Task {task}: {task.__doc__}"
    #     embedding = model.encode(task_desc)
    #     meta_dict = {}
    #     meta_dict["embedding"] = embedding
    #     torch.save(meta_dict, meta_path)

    class SingleTaskDataset:
        def __init__(self, root_path: str) -> None:
            self.files = []
            self.root_path = root_path
            for filename in os.listdir(root_path):
                if filename.startswith("episode") and filename.endswith(".pt"):
                    path = os.path.join(self.root_path, filename)
                    self.files.append(path)

        @property
        def n_episodes(self):
            return len(self.files)

        def clear(self):
            while len(self.files):
                path = self.files.pop()
                os.remove(path)

        def add_episode(self, episode_data: dict):
            len_high = episode_data["episode_len_high"]
            len_low = episode_data["episode_len_low"]
            episode_path = os.path.join(
                dataset_path, f"episode_{self.n_episodes:03}_{len_high}_{len_low}.pt")
            msg = termcolor.colored(f"Save episode data to {episode_path}", "green")
            logging.info(msg)
            torch.save(episode_data, episode_path)
            self.files.append(episode_path)

    dataset = SingleTaskDataset(root_path=dataset_path)
    logging.info(f"There are {dataset.n_episodes} episodes existing.")
    
    if 'regenerate_data' in cfg:
        dataset.n_episodes = 0

    for seed in range(dataset.n_episodes, cfg["n"]):
        logging.info(f"Collecting episode {dataset.n_episodes} with seed {seed}:")
        env.set_task(task)
        obs, info = env.reset(seed)
        try:
            total_reward = 0
            for _ in range(task.max_steps):
                act = agent.act(obs, info)
                assert act is not None
                lang_goal = info['lang_goal']
                obs, reward, done, info = env.step(act)
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                if done:
                    break
                
                imageio.imsave(f"{_:03}.jpg", obs["color"][0])
            episode_data, ims = env.get_episode_data(rgb_array=True)
            gif_path = os.path.join(dataset_path, f"episode_{dataset.n_episodes:03}.gif")
            imageio.mimsave(gif_path, ims, format="gif")
            dataset.add_episode(episode_data)
        except Exception as e:
            logging.error(f"Error: {e}")
            continue
    

if __name__ == '__main__':
    main()

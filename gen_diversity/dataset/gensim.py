import os
import torch
import numpy as np
import einops
import inspect
import cliport
import time
import pybullet as p

from tqdm import tqdm
from torch.utils.data import Dataset
from tensordict import TensorDict, MemoryMappedTensor
from torchvision.transforms.functional import resize
from collections import defaultdict
from typing import Mapping

from cliport import tasks
from cliport.environments.environment import Environment as _Environment

GENSIM_ROOT = os.path.join(cliport.__path__[0], "../")
os.environ["GENSIM_ROOT"] = GENSIM_ROOT
print(GENSIM_ROOT)

def dict_stack(dicts):
    keys = dicts[0].keys()
    result = {}
    for k in keys:
        result[k] = np.stack([d[k] for d in dicts])
    return result

class GenSimEnvironment(_Environment):
    
    decimation: int = 8 # 240 / 8 = 30Hz

    def __init__(self, assets_root, task=None, disp=False, shared_memory=False, hz=240, record_cfg=None):
        super().__init__(assets_root, task, disp, shared_memory, hz, record_cfg)
        # self.agent_cams[0]["image_size"] = (180, 240)
        # self.agent_cams[0]["intrinsics"] = (225, *self.agent_cams[0]["intrinsics"][1:])
        self.agent_cams[0]["image_size"] = (180, 240)
        self.agent_cams[0]["intrinsics"] = (160, *self.agent_cams[0]["intrinsics"][1:])

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_counter = 0
        self._traj = defaultdict(list)
        self._seed = seed
        obs = super().reset()
        info = self.info
        return obs, info
    
    def step(self, action=None):
        if action is not None:
            action_high = {
                "pose0": np.concatenate(action["pose0"]),
                "pose1": np.concatenate(action["pose1"]),
            }
            self._traj["action_high"].append(action_high)
            self._traj["high_step"].append(len(self._traj["action_low"]))
        # else:
        #     # called by `.reset`
        #     # get init state
        #     jstate = p.getJointStates(self.ur5, self.joints)
        #     currj = np.array([state[0] for state in jstate])
        #     currjdot = np.array([state[1] for state in jstate])
        #     rgb, depth, seg = self.render_camera(self.agent_cams[0])
        #     low_obs = {
        #         "state": np.concatenate([currj, currjdot]),
        #         "rgb": rgb,
        #         "depth": einops.rearrange(depth, "h w -> h w 1")
        #     }
        #     self._traj["obs_low"].append(low_obs)

        obs, reward, done, info = super().step(action)
        self._traj["lang_goal"].append(info["lang_goal"])
        self._traj["obs_high"].append({
            "color_0": obs["color"][0],
            "color_1": obs["color"][1],
            "color_2": obs["color"][2],
            "depth_0": obs["depth"][0],
            "depth_1": obs["depth"][1],
            "depth_2": obs["depth"][2],
        })
        
        # joint state
        jstate = p.getJointStates(self.ur5, self.joints)
        currj = np.array([state[0] for state in jstate])
        currjdot = np.array([state[1] for state in jstate])

        self.jstate = np.concatenate([currj, currjdot])
        return obs, reward, done, info
    
    def movej(self, targj, speed=0.01, timeout=150):
        """Move UR5 to target joint configuration."""

        t0 = time.time()
        while (time.time() - t0) < timeout:
            # currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            # currj = np.array(currj)
            jstate = p.getJointStates(self.ur5, self.joints)
            currj = np.array([state[0] for state in jstate])
            currjdot = np.array([state[1] for state in jstate])

            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            
            if self.step_counter % self.decimation == 0:
                rgb, depth, seg = self.render_camera(self.agent_cams[0])
                self._traj["obs_low"].append({
                    "state": np.concatenate([currj, currjdot]),
                    "rgb": rgb,
                    "depth": einops.rearrange(depth, "h w -> h w 1")
                })
                self._traj["action_low"].append(stepj)

            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_simulation()
        
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True
    
    def get_episode_data(self, rgb_array: bool=False):
        episode_len_high = len(self._traj["action_high"])
        episode_len_low = len(self._traj["obs_low"])

        is_terminal = np.zeros(episode_len_low, dtype=bool)
        is_first    = np.zeros(episode_len_low, dtype=bool)
        is_terminal[-1] = True
        is_first[0] = True
        
        obs_high = dict_stack(self._traj["obs_high"])
        obs_high = {k: v[:-1] for k, v in obs_high.items()} # exclude last obs
        action_high = dict_stack(self._traj["action_high"])
        if not len(obs_high["color_0"]) == len(action_high["pose0"]) == episode_len_high:
            raise ValueError("High-level episode length mismatch:"
                            f"obs_high: {len(obs_high['color_0'])},"
                            f"action_high: {len(action_high['pose0'])},"
                            f"episode_len_high: {episode_len_high}")

        
        obs_low = dict_stack(self._traj["obs_low"])
        action_low = np.stack(self._traj["action_low"])
        if not len(obs_low["rgb"]) == len(action_low) == episode_len_low:
            raise ValueError("Low-level episode length mismatch:"
                            f"obs_low: {len(obs_low['rgb'])},"
                            f"action_low: {len(action_low)},"
                            f"episode_len_low: {episode_len_low}")

        episode_data = {
            "obs_high": obs_high,
            "action_high": action_high,
            "obs_low": obs_low,
            "action_low": action_low,
            "high_step": self._traj["high_step"],
            "is_first": is_first,
            "is_terminal": is_terminal,
            "episode_len_high": episode_len_high,
            "episode_len_low": episode_len_low,
            "seed": self._seed,
        }

        if rgb_array:
            return episode_data, obs_high["color_0"]
        else:
            return episode_data


class GensimDataset(Dataset):

    image_size = (180, 240)
    
    def __init__(self, data: TensorDict, seq_length: int, high_level: bool=False):
        super().__init__()
        self.data = data
        self.total_length = data.shape[0]
        self.seq_length = seq_length
        self.high_level = high_level

        self.starts = self.data["is_first"].nonzero().squeeze(-1)
        self.ends = self.data["is_terminal"].nonzero().squeeze(-1)
        assert len(self.starts) == len(self.ends)
        try:
            from cliport.tasks import names
            self.task_desc = {}
            for task_name in self.tasks:
                task_cls = names[task_name]
                self.task_desc[task_name] = f"Task {task_name}: " + inspect.getdoc(task_cls)
        except:
            pass
    
    def __len__(self):
        return self.total_length - self.seq_length + 1

    def __getitem__(self, index):
        if isinstance(index, int):
            data = self.data[index: index+self.seq_length]
            return data
        else:
            data = torch.stack([self[i] for i in index])
            return data

    @classmethod
    def make(
        cls, 
        root_path, 
        seq_length=20, 
        tasks: list[str]=None, 
        high_level: bool=False,
        max_episodes: int=None,
    ):

        if tasks is None:
            tasks = os.listdir(root_path)
        if "memmaped" in tasks:
            tasks.remove("memmaped")
        if "memmaped_low" in tasks:
            tasks.remove("memmaped_low")

        file_paths = []
        episode_lengths_high = []
        episode_lengths_low = []
        for task in tasks:
            task_path = os.path.join(root_path, task)
            file_names = sorted([
                filename for filename in os.listdir(task_path) 
                if (filename.startswith("episode") and filename.endswith(".pt"))
            ])
            if max_episodes is not None:
                file_names = file_names[:max_episodes]

            for file_name in file_names:
                file_paths.append(os.path.join(task_path, file_name))
                split = file_name[:-3].split("_")
                episode_len_high = int(split[-2])
                episode_len_low = int(split[-1])
                episode_lengths_high.append(episode_len_high)
                episode_lengths_low.append(episode_len_low)
        
        total_length_high = sum(episode_lengths_high)
        total_length_low = sum(episode_lengths_low)
        
        if high_level:
            data = cls._make_high(file_paths, total_length_high, memmap_path=os.path.join(root_path, f"memmaped_high-{max_episodes}"))
        else:
            data = cls._make_low(file_paths, total_length_low, memmap_path=os.path.join(root_path, f"memmaped_low-{max_episodes}"))

        return cls(data, seq_length, high_level)
    
    @staticmethod
    def _make_low(file_paths, total_length: int, memmap_path):
        print(f"Loading {total_length} low steps from {len(file_paths)} episodes")

        data = TensorDict({
            "image": MemoryMappedTensor.empty(total_length, 4, *GensimDataset.image_size),
            "state": MemoryMappedTensor.empty(total_length, 12),
            "action": MemoryMappedTensor.empty(total_length, 6),
            "episode_id": MemoryMappedTensor.empty(total_length, dtype=torch.int),
            "is_first": MemoryMappedTensor.empty(total_length, dtype=torch.bool),
            "is_terminal": MemoryMappedTensor.empty(total_length, dtype=torch.bool),
        }, [total_length])

        data.memmap_(memmap_path)

        cursor = 0
        for i, file_path in tqdm(enumerate(file_paths)):
            episode_data = torch.load(file_path)
            image_rgb = torch.as_tensor(episode_data["obs_low"]["rgb"])
            image_depth = torch.as_tensor(episode_data["obs_low"]["depth"])
            image = torch.cat([image_rgb, image_depth], dim=-1)
            state = torch.as_tensor(episode_data["obs_low"]["state"])
            action = torch.as_tensor(episode_data["action_low"])
            episode_len = len(image)
            episode_data = TensorDict({
                "image": image.permute(0, 3, 1, 2),
                "state": state,
                "action": action,
                "episode_id": torch.full((episode_len,), i, dtype=torch.int64),
                "is_first": torch.zeros(episode_len, dtype=torch.bool),
                "is_terminal": torch.zeros(episode_len, dtype=torch.bool),
            }, [episode_len])
            episode_data["is_first"][0] = True
            episode_data["is_terminal"][-1] = True

            data[cursor: cursor+episode_len] = episode_data
            del episode_data
            cursor += episode_len

        return data
    
    @staticmethod
    def _make_high(file_paths, total_length: int, memmap_path):
        print(f"Loading {total_length} high steps from {len(file_paths)} episodes")

        data = TensorDict({
            "image": MemoryMappedTensor.empty(total_length, 3, *GensimDataset.image_size),
            "state": MemoryMappedTensor.empty(total_length, 12),
            "action": MemoryMappedTensor.empty(total_length, 14),
            "episode_id": MemoryMappedTensor.empty(total_length, dtype=torch.int),
            "is_first": MemoryMappedTensor.empty(total_length, dtype=torch.bool),
            "is_terminal": MemoryMappedTensor.empty(total_length, dtype=torch.bool),
        }, [total_length])

        data.memmap_(memmap_path)

        cursor = 0
        for i, file_path in enumerate(tqdm(file_paths)):
            episode_data = torch.load(file_path)
            image = torch.as_tensor(episode_data["obs_high"]["color_0"])
            state = torch.as_tensor(episode_data["obs_low"]["state"])
            state = state[episode_data["high_step"]]
            
            action = episode_data["action_high"]
            action = np.concatenate([action["pose0"], action["pose1"]], axis=1)
            
            episode_len = len(image)
            episode_data = TensorDict({
                "image": image.permute(0, 3, 1, 2),
                "state": state,
                "action": torch.as_tensor(action),
                "episode_id": torch.full((episode_len,), i, dtype=torch.int64),
                "is_first": torch.zeros(episode_len, dtype=torch.bool),
                "is_terminal": torch.zeros(episode_len, dtype=torch.bool),
            }, [episode_len])
            episode_data["is_first"][0] = True
            episode_data["is_terminal"][-1] = True

            data[cursor: cursor+episode_len] = episode_data
            del episode_data
            cursor += episode_len
        
        # if not cursor == total_length:
        #     raise ValueError(f"Expected {total_length} but got {cursor}")

        return data


    @classmethod
    def load(cls, root_path, seq_length=20, high_level: bool=False, max_episodes: int=None):
        if high_level:
            data = TensorDict.load_memmap(os.path.join(root_path, f"memmaped_high-{max_episodes}"))
        else:
            data = TensorDict.load_memmap(os.path.join(root_path, f"memmaped_low-{max_episodes}"))
        return cls(data, seq_length, high_level)

if __name__ == "__main__":
    GensimDataset.make("/home/btx0424/gensim_ws/GenSim/data/train", high_level=True)
    GensimDataset.load("/home/btx0424/gensim_ws/GenSim/data/train", high_level=True)

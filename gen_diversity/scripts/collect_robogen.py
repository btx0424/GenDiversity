import yaml
import torch
import pickle
import imageio
import einops
import os
import os.path as osp
import sys
import numpy as np
import pybullet as p
import time, datetime
import json
import itertools
from torchvision.transforms.functional import resize

FILE_PATH = osp.dirname(osp.abspath(__file__))
IMAGE_SIZE = (192, 240)
ROBOGEN_PATH = "/media/aa/hdd1/EvalTask/ext/RoboGen"

os.chdir(ROBOGEN_PATH)
sys.path.append(ROBOGEN_PATH)
from RL.ray_learn import load_policy, make_env, tune
from manipulation.utils import save_numpy_as_gif, save_env, take_round_images, build_up_env, load_gif


def process_ims(ims: list) -> np.ndarray:
    ims = torch.as_tensor(np.array(ims))
    ims = einops.rearrange(ims, "t h w c -> t c h w")
    ims = resize(ims, IMAGE_SIZE)
    ims = einops.rearrange(ims, "t c h w -> t h w c")
    return ims


def find_ckpt_path(solution_path):
    if solution_path is None:
        return None, None, None, None
    
    all_substeps = os.path.join(solution_path, "substeps.txt")
    with open(all_substeps, 'r') as f:
        substeps = [line.strip() for line in f.readlines()]

    substep_types = os.path.join(solution_path, "substep_types.txt")
    with open(substep_types, 'r') as f:
        substep_types = [line.strip() for line in f.readlines()]

    action_spaces = os.path.join(solution_path, "action_spaces.txt")
    with open(action_spaces, 'r') as f:
        action_spaces = [line.strip() for line in f.readlines()]

    # print("all substeps:\n {}".format("".join(substeps)))
    # print("all substep types:\n {}".format("".join(substep_types)))

    policy_path = os.path.join(solution_path, "RL_sac")
    if not os.path.exists(policy_path):
        return None, None, None, None
    
    # find policy path
    checkpoints = sorted(os.listdir(policy_path))
    if len (checkpoints) == 0:
        ckpt_path = None
        last_restore_state_file = None
    else:
        found_ckpt = False
        for time_str in checkpoints:
            for i, substep in enumerate(substeps):
                if substep_types[i] == "reward":
                    step_name = substep.strip().replace(" ", "_")
                    ckpt_path = os.path.join(policy_path, time_str, step_name, "best_model")
                    print("ckpt_path: ", ckpt_path)
                    if not os.path.exists(ckpt_path):
                        continue
                    # find the latest checkpoint
                    latest = sorted(os.listdir(ckpt_path))[-1]
                    num = int(latest.split("_")[-1])
                    ckpt_path = os.path.join(ckpt_path, latest, f"checkpoint-{num}")
                    assert os.path.exists(ckpt_path), f"ckpt_path {ckpt_path} does not exist."
                    
                    prev_step_name = substeps[i-1].strip().replace(" ", "_")
                    state_dir = os.path.join(solution_path, "primitive_states", time_str, prev_step_name)
                    state_paths = [path for path in os.listdir(state_dir) if path.endswith(".pkl")]
                    last_restore_state_file = os.path.join(state_dir, sorted(state_paths, key=lambda name: int(name[:-4].split("_")[-1]))[-1])
                    
                    action_space = action_spaces[i]
                    found_ckpt = True
                    break
            if found_ckpt:
                break
    if not found_ckpt:
        ckpt_path = None
        step_name = None
        action_space = None
        last_restore_state_file = None
    return ckpt_path, step_name, action_space, last_restore_state_file


def execute(
    task_config_path, 
    time_string=None, resume=False, # these two are combined for resume training.
    gui=False, 
    randomize=False, # whether to randomize the initial state of the environment.
    use_bard=True, # whether to use the bard to verify the retrieved objects.
    use_gpt_size=True, # whether to use the size from gpt.
    use_gpt_joint_angle=True, # whether to initialize the joint angle from gpt.
    use_gpt_spatial_relationship=True, # whether to use the spatial relationship from gpt.
    obj_id=0, # which object to use from the list of possible objects.
    use_motion_planning=True,
    use_distractor=False,
    skip=[], # which substeps to skip.
    move_robot=False, # whether to move the robot to the initial state.
    only_learn_substep=None,
    last_restore_state_file=None,
):
    with open(task_config_path, 'r') as file:
        task_config = yaml.safe_load(file)
    solution_path = None
    for obj in task_config:
        if "solution_path" in obj:
            solution_path = obj["solution_path"]
            break
    
    dataset_path = "/media/aa/hdd1/cf/data/train/"
    
    whole_task_name = task_config[5]["task_name"]
    whole_task_data_path = os.path.join(dataset_path, whole_task_name.replace(" ", "_"))
    os.makedirs(whole_task_data_path, exist_ok=True)
    
    files = []
    for filename in os.listdir(whole_task_data_path):
        if filename.endswith(".pt"):
            path = os.path.join(whole_task_data_path, filename)
            files.append(path)
    n_episodes = len(files)
    
    ckpt_path, step_name, action_space, last_restore_state_file = find_ckpt_path(solution_path)
    print("ckpt_path: ", ckpt_path, "last_restore_state_file: ", last_restore_state_file)
    # input("Press Enter to continue...")
    print("Start collecting data for task: ", whole_task_name)
    print("There are {}/40 episodes.".format(n_episodes))
    if n_episodes >= 40:
        print("Already collected 40 episodes for this task. Skip.")
        return

    task_name = step_name
    env_config = {
        "task_config_path": task_config_path,
        "solution_path": solution_path,
        "task_name": task_name,
        "last_restore_state_file": last_restore_state_file,
        "action_space": action_space,
        "randomize": True,
        "use_bard": use_bard,
        "obj_id": obj_id,
        "use_gpt_size": use_gpt_size,
        "use_gpt_joint_angle": use_gpt_joint_angle,
        "use_gpt_spatial_relationship": use_gpt_spatial_relationship,
        "use_distractor": use_distractor
    }
    tune.register_env(task_name, lambda config: make_env(env_config))
    policy, _ = load_policy("sac", env_name=task_name, policy_path=ckpt_path, env_config=env_config, seed=2)
    
    env = make_env(env_config)
    for ep_id in range(n_episodes, 40):
        obs = env.reset()
        done = False
        reward_total = 0
        rgbs = []

        actions = []
        observations = []

        for i in itertools.count():
            observations.append(obs)
            action = policy.compute_action(obs)
            obs, reward, done, info = env.step(action)
            rgb, depth = env.render("rgb_array")
            reward_total += reward
            
            rgbs.append(rgb)
            actions.append(action)

            if done:
                break
    
        episode_len  = len(rgbs)
        print("reward_total: ", reward_total, "episode length: ", episode_len)
        
        is_first = np.zeros(episode_len, dtype=bool)
        is_terminal = np.zeros(episode_len, dtype=bool)
        is_terminal[-1] = True
        is_first[0] = True

        ims = process_ims(rgbs)

        episode_data = {
            "action": np.stack(actions),
            "state": np.stack(observations),
            "image": ims,
            "is_first": is_first,
            "is_terminal": is_terminal,
            "length": episode_len
        }

        episode_path = os.path.join(whole_task_data_path, f"episode_{ep_id}_{episode_len}.pt")
        torch.save(episode_data, episode_path)
        episode_gif_path = episode_path.replace(".pt", ".mp4")
        imageio.mimsave(episode_gif_path, ims, format="mp4")
        print(f"Save video to {episode_gif_path}.")

    env.disconnect()


def check_tasks(root_dir):
    task_config_paths = {}
    for task_name in os.listdir(root_dir):
        if str.islower(task_name[0]):
            continue
        for name in os.listdir(os.path.join(root_dir, task_name)):
            if name.startswith(task_name):
                task_config_path = os.path.join(root_dir, task_name, name)
        with open(task_config_path, 'r') as file:
            task_config = yaml.safe_load(file)
        solution_path = None
        for obj in task_config:
            if "solution_path" in obj:
                solution_path = obj["solution_path"]
                break
        ckpt_path = find_ckpt_path(solution_path)[0]
        if ckpt_path is not None:
            task_config_paths[task_name] = task_config_path
    
    yaml.dump(task_config_paths, open("task_config_paths.yaml", "w"))
    return task_config_paths
        
import random
task_config_paths = check_tasks(os.path.join(ROBOGEN_PATH, "example_tasks"))
task_config_paths = dict(sorted(task_config_paths.items(), key=lambda item: random.random()))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config_path', type=str, default=None)
    parser.add_argument('--training_algo', type=str, default="RL_sac")
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--time_string', type=str, default=None)
    parser.add_argument('--gui', type=int, default=0) 
    parser.add_argument('--randomize', type=int, default=0) # whether to randomize roation of objects in the scene.
    parser.add_argument('--obj_id', type=int, default=0) # which object from the list of possible objects to use.
    parser.add_argument('--use_bard', type=int, default=1) # whether to use bard filtered objects.
    parser.add_argument('--use_gpt_size', type=int, default=1) # whether to use size outputted from gpt.
    parser.add_argument('--use_gpt_spatial_relationship', type=int, default=1) # whether to use gpt spatial relationship.
    parser.add_argument('--use_gpt_joint_angle', type=int, default=1) # whether to use initial joint angle output from gpt.
    parser.add_argument('--run_training', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--use_motion_planning', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--use_distractor', type=int, default=1) # if to train or just to build the scene.
    parser.add_argument('--skip', nargs="+", default=[]) # if to train or just to build the scene.
    parser.add_argument('--move_robot', type=int, default=0) # if to train or just to build the scene.
    parser.add_argument('--only_learn_substep', type=int, default=None) # if to run learning for a substep.
    parser.add_argument('--reward_learning_save_path', type=str, default=None) # where to store the learning result of RL training. 
    parser.add_argument('--last_restore_state_file', type=str, default=None) # whether to start from a specific state.
    args = parser.parse_args()

    if args.task_config_path is None:
        for key, value in task_config_paths.items():
            execute(
                value,
                resume=args.resume, 
                time_string=args.time_string, 
                gui=args.gui, 
                randomize=args.randomize,
                use_bard=args.use_bard,
                use_gpt_size=args.use_gpt_size,
                use_gpt_joint_angle=args.use_gpt_joint_angle,
                use_gpt_spatial_relationship=args.use_gpt_spatial_relationship,
                obj_id=args.obj_id,
                use_motion_planning=args.use_motion_planning,
                use_distractor=args.use_distractor,
                skip=args.skip,
                move_robot=args.move_robot,
                only_learn_substep=args.only_learn_substep,
                last_restore_state_file=args.last_restore_state_file
            )
    else:
        execute(
            args.task_config_path, 
            resume=args.resume, 
            time_string=args.time_string, 
            gui=args.gui, 
            randomize=args.randomize,
            use_bard=args.use_bard,
            use_gpt_size=args.use_gpt_size,
            use_gpt_joint_angle=args.use_gpt_joint_angle,
            use_gpt_spatial_relationship=args.use_gpt_spatial_relationship,
            obj_id=args.obj_id,
            use_motion_planning=args.use_motion_planning,
            use_distractor=args.use_distractor,
            skip=args.skip,
            move_robot=args.move_robot,
            only_learn_substep=args.only_learn_substep,
            last_restore_state_file=args.last_restore_state_file
        )
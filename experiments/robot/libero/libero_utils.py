"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import os.path as osp

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(
    rollout_images,
    rollout_wrist_images,
    idx,
    success,
    task_description,
    log_dir,
    log_file=None,
    save_image=False,
):
    """Saves an MP4 replay of an episode, combining main and wrist camera views."""
    rollout_dir = osp.join(log_dir, f"rollouts/{DATE}")
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)

    if save_image:
        dir_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}"
        os.makedirs(dir_path)

    if rollout_wrist_images:
        for frame_idx, (main_img, wrist_img) in enumerate(zip(rollout_images, rollout_wrist_images)):
            if save_image:
                main_file_name = f"{dir_path}/primary--frame={frame_idx:04d}.bmp"
                imageio.imwrite(main_file_name, main_img)
                wrist_file_name = f"{dir_path}/wrist--frame={frame_idx:04d}.bmp"
                imageio.imwrite(wrist_file_name, wrist_img)

            h1, w1, _ = main_img.shape
            h2, w2, _ = wrist_img.shape
            if h1 != h2:
                target_h = max(h1, h2)
                main_img = np.array(Image.fromarray(main_img).resize((int(w1 * target_h / h1), target_h)))
                wrist_img = np.array(Image.fromarray(wrist_img).resize((int(w2 * target_h / h2), target_h)))

            combined_img = np.concatenate((main_img, wrist_img), axis=1)
            video_writer.append_data(combined_img)
    else:
        for img in rollout_images:
            video_writer.append_data(img)

    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

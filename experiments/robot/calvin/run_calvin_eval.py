"""Code to evaluate Calvin."""

import copy
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import draccus
import hydra
import imageio
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm
from experiments.robot.calvin.vla_evaluation import (
    OPENVLA_IMAGE_SIZE,
    DualSystemCalvinEvaluation,
    center_crop_image,
    check_image_format,
    prepare_policy_image,
)

from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    get_vision_attn_weight_generator,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_model,
)
from prismatic.attention_viz import LAYER_GROUPS, save_attention_snapshot

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ["CALVIN_ROOT"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CALVIN_DATASET_REL_PATH = "dataset/task_ABC_D"
DEFAULT_EP_LEN = 360
DEFAULT_NUM_SEQUENCES = 1000
MAX_ROLLOUT_STEPS = 800
OPEN_LOOP_STEPS = 8
VIDEO_FPS = 50
VIDEO_BITRATE = "5000k"


@dataclass
class MultiFrameConfig:
    # fmt: off
    temporal_strategy: Optional[str] = None          # Temporal conditioning strategy: ["attn_weight"]
    temporal_feature_source: Optional[str] = None    # Temporal feature source: ["action"]
    temporal_feature_layer: int = -2                 # Index of hidden states of LLM. -1: last hidden state with norm, -2: last hidden state without norm, ...

    # Configs of "attn_weight" temporal_strategy
    # force_eager_attn = True, extra_layer_ids = (layer_i, layer_j, ...):
    #     The layers in the extra_layer_ids will use LlamaAttention instead of LlamaSdpaAttention,
    #     and the extra_attn_weights will be used to modulate the softmax operation with the format
    #     of e^(x_i) * weight_i.
    attn_weight_force_eager_attn: bool = False
    attn_weight_extra_layer_ids: Optional[str] = None  # String of indexes. (e.g. "", "(1, 2, 3)", "(i for i in range(0, 32))")
    attn_weight_head_type: str = "softmaxscore"      # "ste", "gumbel", "softmaxscore", "sigmoidscore"
    attn_weight_score_config: str = "(1.9, 0.1, 0.0)"  # (pos_score, neg_score, score_bias) for softmaxscore
    attn_weight_sink_ids: list[int] = None           # Indexes of image patches to force in extra_attn_weights
    attn_weight_sink_weight: float = 1               # Forced value
    attn_weight_attn_ratio: Optional[float] = None   # The value to implement mean constraint of vision_attn_weight. (mean = attn_ratio)

    # fmt: on


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "../outputs/calvin-abc"     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    use_x0_prediction: bool = False
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # CALVIN
    #################################################################################################################
    enrich_lang: bool = False
    single_task: Optional[str] = None                # Evaluate one task once instead of 1000 five-task sequences
    task_sequence: Optional[str] = None              # Comma-separated five-task CALVIN sequence

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    seed: int = 7                                    # Random Seed (for reproducibility)

    # Temporal configs (optional)
    multi_frame: MultiFrameConfig = field(default_factory=MultiFrameConfig)
    vis_mask: bool = False
    attention_viz: bool = False                      # Save raw/final attention matrices during CALVIN inference
    attention_viz_dir: str = "./experiments/attention_viz_calvin"
    attention_layer_group: str = "L0-L31"
    raw_attention_eval: bool = False                 # Disable AVA visual attention reweighting during evaluation
    disable_ava_module: bool = False                 # Disable AVA generator and its temporal conditioning

    save_version: str = "Pro"                        # version of exps

    # fmt: on


def print_and_save(results, sequences, eval_result_path, task_name=None, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    if not os.path.isdir(f"./{task_name}"):
        os.mkdir(f"./{task_name}")
    with open(f"./{task_name}/split_{torch.cuda.current_device()}.json", "w") as file:
        json.dump(chain_sr, file)

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from experiments.robot.calvin.calvin_env_wrapper import CalvinEnvWrapperRaw

    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def load_validation_annotations(conf_dir: Path, enrich_lang: bool):
    if enrich_lang:
        with open(Path(__file__).with_name("enrich_lang_annotations.json"), "r") as f:
            return json.load(f)
    return OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")


def get_process_sequences(num_sequences: int, num_procs: int, procs_id: int):
    eval_sequences = get_sequences(num_sequences)
    num_seq_per_procs = num_sequences // num_procs
    return eval_sequences[num_seq_per_procs * procs_id : num_seq_per_procs * (procs_id + 1)]


def append_success_rates(eval_sr_path, sequence_i: int, num_sequences: int, success_list) -> None:
    line = f"{sequence_i}/{num_sequences}: "
    for sr in success_list:
        line += f"{sr:.3f} | "
    line += "\n"
    with open(eval_sr_path, "a") as f:
        f.write(line)


def evaluate_policy(
    cfg,
    model,
    env,
    eval_sr_path,
    eval_result_path,
    num_procs,
    procs_id,
    eval_dir,
    ep_len,
    num_sequences,
    task_name="test",
    enrich_lang=False,
    debug=False,
):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    val_annotations = load_validation_annotations(conf_dir, enrich_lang)
    eval_dir = get_log_dir(eval_dir)
    if cfg.single_task is not None and cfg.task_sequence is not None:
        raise ValueError("single_task and task_sequence are mutually exclusive.")

    if cfg.task_sequence is not None:
        selected_tasks = tuple(task.strip() for task in cfg.task_sequence.split(",") if task.strip())
        if len(selected_tasks) != 5:
            raise ValueError("task_sequence must contain exactly five comma-separated task names.")
        unknown_tasks = [task for task in selected_tasks if task not in val_annotations]
        if unknown_tasks:
            raise ValueError(f"Unknown task_sequence entries: {unknown_tasks}")
        if num_procs != 1:
            raise ValueError("task_sequence evaluation must run with one process; launch CALVIN on one GPU/process.")
        initial_state, _ = get_sequences(1)[0]
        eval_sequences = [(initial_state, selected_tasks)]
        eval_num_sequences = 1
    elif cfg.single_task is not None:
        if cfg.single_task not in val_annotations:
            available_tasks = sorted(val_annotations.keys())
            raise ValueError(
                f"Unknown single_task {cfg.single_task!r}; available tasks include {available_tasks[:10]}"
            )
        if num_procs != 1:
            raise ValueError("single_task evaluation must run with one process; launch CALVIN on one GPU/process.")
        initial_state, _ = get_sequences(1)[0]
        eval_sequences = [(initial_state, (cfg.single_task,))]
        eval_num_sequences = 1
    else:
        eval_sequences = get_process_sequences(num_sequences, num_procs, procs_id)
        eval_num_sequences = num_sequences

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            cfg,
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_dir,
            sequence_i,
            ep_len,
        )
        results.append(result)
        if not debug:
            success_list = count_success(results)
            append_success_rates(eval_sr_path, sequence_i, eval_num_sequences, success_list)
            sequence_i += 1
            eval_sequences.set_description(
                " ".join([f"{i + 1}/{len(eval_sequence)} : {v * 100:.1f}% |" for i, v in enumerate(success_list)])
                + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, task_name, None)
    return results


def evaluate_sequence(
    cfg,
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug,
    eval_dir,
    sequence_i,
    ep_len,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout_hi3(
            cfg, env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len
        )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        # normalized_action[..., -1] = np.sign(normalized_action[..., -1])
        sign = np.sign(normalized_action[..., -1])
        sign = np.array(sign)  # Ensure it is an array and not a scalar
        sign[sign == 0.0] = 1  # Change 0 to 1
        sign[sign == -0.0] = -1  # Change -0 to -1
        normalized_action[..., -1] = sign

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def vis(primary_image, wrist_image, vision_attn_weights, only_primary=False, is_continuous=True, center_crop=True):
    import matplotlib.pyplot as plt

    def prepare_images_for_vis(images: List[np.ndarray]) -> List[Image.Image]:
        processed_images = []

        for image in images:
            check_image_format(image)

            if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
                image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

            pil_image = Image.fromarray(image).convert("RGB")
            if center_crop:
                pil_image = center_crop_image(pil_image)

            processed_images.append(np.array(pil_image))

        return processed_images

    def apply_mask_to_image(image_array, mask_array, min_val=None, max_val=None, is_continuous=True):
        assert image_array.ndim == 3 and image_array.shape[-1] == 3  # (H, W, C)
        assert mask_array.ndim == 2  # (h, w)

        min_val = np.min(mask_array) if min_val is None else min_val
        max_val = np.max(mask_array) if max_val is None else max_val
        max_val = max_val if max_val > min_val else min_val + 1e-12

        scale_factor_h = image_array.shape[0] // mask_array.shape[0]
        scale_factor_w = image_array.shape[1] // mask_array.shape[1]
        mask_resized = np.repeat(np.repeat(mask_array, scale_factor_h, axis=0), scale_factor_w, axis=1)

        result = image_array.copy().astype(np.float32)
        if not is_continuous:
            alpha = 0.5
            mask_color = (0, 0, 255)

            colored_mask = np.zeros_like(image_array)
            colored_mask[:, :, 0] = mask_color[0]  # R
            colored_mask[:, :, 1] = mask_color[1]  # G
            colored_mask[:, :, 2] = mask_color[2]  # B

            mask_3channel = np.stack([mask_resized, mask_resized, mask_resized], axis=2)
            mask_areas = mask_3channel > 0.5
            result[mask_areas] = (1 - alpha) * result[mask_areas] + alpha * colored_mask[mask_areas]
        else:
            alpha = 0.3
            cmap = plt.get_cmap("viridis")

            mask_normalized = np.clip((mask_resized - min_val) / (max_val - min_val), 0, 1)
            heatmap_colors = cmap(mask_normalized)  # (224, 224, 4) RGBA
            heatmap_rgb = heatmap_colors[:, :, :3]
            heatmap_rgb = heatmap_rgb * 255

            result = (1 - alpha) * result + alpha * heatmap_rgb

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    primary_image, wrist_image = prepare_images_for_vis([primary_image, wrist_image])

    N = vision_attn_weights.shape[1]
    primary_end_idx = N if only_primary else N // 2

    primary_image_mask = vision_attn_weights[:, :primary_end_idx].detach()
    primary_image_mask = primary_image_mask.reshape(16, 16).to(torch.float32).cpu().numpy()  # bs == 1
    primary_image = apply_mask_to_image(primary_image, primary_image_mask, is_continuous=is_continuous)

    if not only_primary:
        writst_image_mask = vision_attn_weights[:, primary_end_idx:].detach()
        writst_image_mask = writst_image_mask.reshape(16, 16).to(torch.float32).cpu().numpy()
        wrist_image = apply_mask_to_image(wrist_image, writst_image_mask, is_continuous=is_continuous)

    return primary_image, wrist_image


def write_video(img_dict, eval_dir, sequence_i, subtask_i, subtask, is_failed=False, only_failed=True, save_image=False):
    if is_failed:
        print()
        print(colored(f"fail {sequence_i}-{subtask_i}-{subtask}", "red"))
    else:
        print(colored("success", "green"), end=" ")

    if only_failed and not is_failed:
        return

    if save_image:
        dir_path = os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{"fail" if is_failed else "succ"}')
        os.makedirs(dir_path)

    for key in img_dict.keys():
        if save_image:
            for idx, img in enumerate(img_dict[key]):
                file_name = f"{dir_path}/{key}--frame={idx:04d}.bmp"
                imageio.imwrite(file_name, img)

        clip = ImageSequenceClip(img_dict[key], fps=VIDEO_FPS)
        clip.write_videofile(
            os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-{"fail" if is_failed else "succ"}.mp4'),
            fps=VIDEO_FPS,
            codec="libx264",
            bitrate=VIDEO_BITRATE,
        )


def init_image_buffer():
    return {
        "static": [],
        "gripper": [],
    }


def append_rollout_images(cfg, img_dict, obs, vision_attn_weights):
    primary_img = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
    wrist_img = copy.deepcopy(obs["rgb_obs"]["rgb_gripper"])
    if cfg.vis_mask and vision_attn_weights is not None:
        primary_img, wrist_img = vis(primary_img, wrist_img, vision_attn_weights, center_crop=cfg.center_crop)
    img_dict["static"].append(primary_img)
    img_dict["gripper"].append(wrist_img)


def single_inference_test(
    cfg,
    model,
    obs,
    lang_annotation,
    env,
    img_dict,
    task_oracle,
    start_info,
    subtask,
    eval_dir,
    sequence_i,
    subtask_i,
):
    temporal_context = None
    for query_idx in range(MAX_ROLLOUT_STEPS // OPEN_LOOP_STEPS):
        action_buffers, temporal_context, vision_attn_weights, attention_snapshot = model.step(
            obs, lang_annotation, temporal_context
        )

        if cfg.attention_viz:
            primary_image = prepare_policy_image(obs["rgb_obs"]["rgb_static"], cfg.center_crop)
            wrist_image = prepare_policy_image(obs["rgb_obs"]["rgb_gripper"], cfg.center_crop)
            save_attention_snapshot(
                attention_snapshot,
                Path(cfg.attention_viz_dir)
                / f"sequence_{sequence_i:04d}"
                / f"subtask_{subtask_i:02d}",
                f"step_{query_idx:04d}",
                primary_image=primary_image,
                wrist_image=wrist_image,
                metadata={
                    "sequence_i": sequence_i,
                    "subtask_i": subtask_i,
                    "subtask": subtask,
                    "query_idx": query_idx,
                    "instruction": lang_annotation,
                    "open_loop_steps": len(action_buffers),
                },
            )
        for action in action_buffers:
            action = process_action(action, "openvla")
            obs, _reward, _done, current_info = env.step(action.tolist())

            append_rollout_images(cfg, img_dict, obs, vision_attn_weights)

            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                write_video(img_dict, eval_dir, sequence_i, subtask_i, subtask)
                return True

    write_video(img_dict, eval_dir, sequence_i, subtask_i, subtask, is_failed=True)
    return False


def rollout_hi3(
    cfg,
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    eval_dir,
    subtask_i,
    sequence_i,
    ep_len,
):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    img_dict = init_image_buffer()

    ret_code = single_inference_test(
        cfg,
        model,
        obs,
        lang_annotation,
        env,
        img_dict,
        task_oracle,
        start_info,
        subtask,
        eval_dir,
        sequence_i,
        subtask_i,
    )

    return ret_code


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    if cfg.raw_attention_eval:
        if not cfg.use_l1_regression or cfg.use_diffusion:
            raise ValueError("raw_attention_eval currently supports CALVIN L1 regression only.")
        cfg.multi_frame.temporal_strategy = "attn_weight"
        cfg.multi_frame.temporal_feature_source = "action"

    if cfg.attention_viz:
        if not cfg.use_l1_regression or cfg.use_diffusion:
            raise ValueError("attention_viz currently supports CALVIN L1 regression only.")
        if cfg.raw_attention_eval:
            raise ValueError("raw_attention_eval and attention_viz cannot be enabled together.")
        if cfg.attention_layer_group not in LAYER_GROUPS:
            raise ValueError(
                f"Unknown attention_layer_group {cfg.attention_layer_group}; choose from {sorted(LAYER_GROUPS)}"
            )
        cfg.multi_frame.temporal_strategy = "attn_weight"
        cfg.multi_frame.temporal_feature_source = "action"
        cfg.multi_frame.attn_weight_force_eager_attn = True
        cfg.multi_frame.attn_weight_extra_layer_ids = "(i for i in range(0, 32))"

    if cfg.disable_ava_module:
        if cfg.raw_attention_eval:
            raise ValueError("raw_attention_eval and disable_ava_module cannot be enabled together.")
        cfg.multi_frame.temporal_strategy = None
        cfg.multi_frame.temporal_feature_source = None


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    if hasattr(model, "set_version"):
        model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    vision_attn_weight_generator = None
    if cfg.multi_frame.temporal_strategy == "attn_weight" and not cfg.disable_ava_module:
        vision_attn_weight_generator = get_vision_attn_weight_generator(
            cfg, model.llm_dim, int(np.sqrt(model.vision_backbone.get_num_patches()))
        )

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    return (
        model,
        action_head,
        proprio_projector,
        noisy_action_projector,
        vision_attn_weight_generator,
        processor,
    )


def get_observation_space():
    return {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }


def get_eval_dir(cfg: GenerateConfig, current_time: str) -> str:
    checkpoint_name = cfg.pretrained_checkpoint.split("/")[-1]
    return cfg.local_log_dir + f"/calvin/{current_time}_{checkpoint_name}/"


def create_calvin_evaluator(
    cfg,
    model,
    proprio_projector,
    noisy_action_projector,
    action_head,
    processor,
    vision_attn_weight_generator,
):
    return DualSystemCalvinEvaluation(
        model,
        proprio_projector,
        noisy_action_projector,
        action_head,
        processor,
        use_x0_prediction=cfg.use_x0_prediction,
        unnorm_key=cfg.unnorm_key,
        center_crop=cfg.center_crop,
        temporal_feature_source=cfg.multi_frame.temporal_feature_source,
        temporal_feature_layer=cfg.multi_frame.temporal_feature_layer,
        vision_attn_weight_generator=vision_attn_weight_generator,
        vision_attn_ratio=cfg.multi_frame.attn_weight_attn_ratio,
        disable_vision_attn_weights=cfg.raw_attention_eval,
        attention_viz=cfg.attention_viz,
        attention_layer_group=cfg.attention_layer_group,
    )


@draccus.wrap()
def main(cfg: GenerateConfig):
    seed_everything(cfg.seed)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    validate_config(cfg)

    model, action_head, proprio_projector, noisy_action_projector, vision_attn_weight_generator, processor = (
        initialize_model(cfg)
    )

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    observation_space = get_observation_space()
    eval_dir = get_eval_dir(cfg, current_time)
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env(os.path.join(CALVIN_ROOT, CALVIN_DATASET_REL_PATH), observation_space, DEVICE)

    eva = create_calvin_evaluator(
        cfg,
        model,
        proprio_projector,
        noisy_action_projector,
        action_head,
        processor,
        vision_attn_weight_generator,
    )
    avg_reward = (
        torch.tensor(
            evaluate_policy(
                cfg,
                eva,
                env,
                eval_dir + "success_rate.txt",
                eval_dir + "result.txt",
                acc.num_processes,
                acc.process_index,
                eval_dir=eval_dir,
                ep_len=DEFAULT_EP_LEN,
                num_sequences=DEFAULT_NUM_SEQUENCES,
                enrich_lang=cfg.enrich_lang,
                debug=False,
            )
        )
        .float()
        .mean()
        .to(DEVICE)
    )

    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print("average success rate ", avg_reward)


if __name__ == "__main__":
    main()

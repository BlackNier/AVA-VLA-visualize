from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from PIL import Image

from prismatic.util.temporal_context_utils import TemporalFeatureExtractor
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

OPENVLA_IMAGE_SIZE = 224


def get_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = crop_and_resize(image, crop_scale, batch_size)

    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def prepare_policy_image(image: np.ndarray, center_crop: bool) -> Image.Image:
    check_image_format(image)
    if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
        image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

    pil_image = Image.fromarray(image).convert("RGB")
    if center_crop:
        pil_image = center_crop_image(pil_image)
    return pil_image


class DualSystemCalvinEvaluation(CalvinBaseModel):
    def __init__(
        self,
        model,
        proprio_projector,
        noisy_action_projector,
        action_head,
        processor,
        use_x0_prediction=False,
        unnorm_key: str = "calvin_abc",
        center_crop: bool = True,
        temporal_feature_source: Optional[str] = None,
        temporal_feature_layer: int = -2,
        vision_attn_weight_generator=None,
        vision_attn_ratio: Optional[float] = None,
    ):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = processor
        self.OFT = model
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.action_head = action_head
        self.unnorm_key = unnorm_key
        self.center_crop = center_crop
        self.vision_attn_weight_generator = vision_attn_weight_generator
        self.vision_attn_ratio = vision_attn_ratio

        self.temporal_feature_source = temporal_feature_source
        self.temporal_feature_layer = temporal_feature_layer
        self.temporal_extractor = TemporalFeatureExtractor()

        if self.action_head is not None and hasattr(self.action_head, "use_x0_prediction"):
            self.action_head.use_x0_prediction = use_x0_prediction

    def reset(
        self,
    ):
        pass

    def _prepare_images(self, obs):
        primary_image = prepare_policy_image(obs["rgb_obs"]["rgb_static"], self.center_crop)
        gripper_image = prepare_policy_image(obs["rgb_obs"]["rgb_gripper"], self.center_crop)
        return primary_image, gripper_image

    def _build_inputs(self, prompt, primary_image, gripper_image):
        inputs = self.processor(prompt, primary_image).to(self.OFT.device, dtype=torch.bfloat16)
        all_wrist_inputs = [self.processor(prompt, [gripper_image]).to(self.OFT.device, dtype=torch.bfloat16)]

        primary_pixel_values = inputs["pixel_values"]
        all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
        inputs["pixel_values"] = torch.cat([primary_pixel_values, *all_wrist_pixel_values], dim=1)
        return inputs

    def _get_normalized_proprio(self, obs):
        proprio_state = np.concatenate([obs["robot_obs"][:7], obs["robot_obs"][-1:]])
        proprio_norm_stats = self.OFT.norm_stats[self.unnorm_key]["proprio"]
        obs["state"] = normalize_proprio(proprio_state, proprio_norm_stats)
        return obs["state"]

    def _get_temporal_context(self, temporal_feature_source, actions_hidden_states, last_hidden_states, input_ids):
        if temporal_feature_source is None:
            return None
        if temporal_feature_source == "action":
            return actions_hidden_states

        frame_batch = {"input_ids": input_ids}
        return self.temporal_extractor.extract_temporal_features(
            temporal_feature_source=temporal_feature_source,
            last_hidden_states=last_hidden_states,
            frame_batch=frame_batch,
            vla_model=self.OFT,
        )

    def step(self, obs, instruction, temporal_context=None):
        """
        Args:
            obs: environment observations
            instruction: embedded language goal
        Returns:
            action: predicted action
        """
        primary_image, gripper_image = self._prepare_images(obs)
        prompt = get_openvla_prompt(instruction)
        inputs = self._build_inputs(prompt, primary_image, gripper_image)
        proprio_state = self._get_normalized_proprio(obs)

        with torch.no_grad():
            pred_outputs = self.OFT.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
                proprio=proprio_state,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                noisy_action_projector=self.noisy_action_projector,
                use_film=False,
                temporal_context=temporal_context,
                temporal_feature_layer=self.temporal_feature_layer,
                vision_attn_weight_generator=self.vision_attn_weight_generator,
                vision_attn_ratio=self.vision_attn_ratio,
            )

            action, actions_hidden_states, last_hidden_states, input_ids, vision_attn_weights = pred_outputs

            temporal_context = self._get_temporal_context(
                self.temporal_feature_source,
                actions_hidden_states,
                last_hidden_states,
                input_ids,
            )

        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()
        action[:, -1] = 1 - action[:, -1]

        return [action[i] for i in range(len(action))], temporal_context, vision_attn_weights

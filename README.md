<h1 align="center">AVA-VLA</h1>

<h3 align="center">Improving Vision-Language-Action Models with Active Visual Attention</h3>

<p align="center" style="margin-bottom: 4px;">
  <a href="https://liauto-dsr.github.io/AVA-VLA-Page/">
    <img src="https://img.shields.io/badge/Project-Page-3b82f6?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2511.18960">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://github.com/LiAuto-DSR/AVA-VLA">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code">
  </a>
  <a href="https://huggingface.co/LiAuto-DSR">
    <img src="https://img.shields.io/badge/Models-Hugging%20Face-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=white" alt="Models">
  </a>
</p>

<p align="center" style="margin-top: 0;">
  <a href="https://cvpr.thecvf.com/virtual/2026/poster/38800">
    <img src="https://img.shields.io/badge/CVPR%202026-Highlight-8b5cf6?style=for-the-badge" alt="CVPR 2026 Highlight">
  </a>
</p>

AVA-VLA improves vision-language-action policy learning by conditioning action generation on a recurrent state that summarizes task history. Built on this temporal state, the Active Visual Attention (AVA) module dynamically reweights visual tokens in the current observation according to the instruction and execution history, helping the policy focus on task-relevant regions in partially observable manipulation tasks.

This repository is based on [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and contains evaluation support for LIBERO and CALVIN.

## LIBERO Results

Success rates (%) on the LIBERO benchmark.

### One Policy for All 4 Suites

| Method | Spatial | Object | Goal | Long | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| pi0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| pi0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| UnifiedVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| OpenVLA-OFT | 97.7 | 98.0 | 96.1 | 95.3 | 96.8 |
| **AVA-VLA** | **97.4** | **99.4** | **97.4** | **97.6** | **98.0** |

### One Policy per Suite

| Method | Spatial | Object | Goal | Long | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| NORA | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| PD-VLA | 95.5 | 96.7 | 94.9 | 91.7 | 94.7 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| FLOWER | 97.5 | 99.1 | 96.1 | 94.9 | 96.9 |
| VLA-Adapter | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| RIPT-VLA | 99.0 | 98.6 | 98.6 | 93.8 | 97.5 |
| **AVA-VLA** | **99.2** | **99.6** | **97.9** | **96.2** | **98.2** |

## CALVIN Results

Success rates (%) on CALVIN ABC -> D. Average length is the mean number of consecutive completed tasks.

| Method | 1 | 2 | 3 | 4 | 5 | Avg. len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| UnifiedVLA | 98.9 | 94.8 | 89.0 | 82.8 | 75.1 | 4.41 |
| OpenVLA-OFT | 96.9 | 92.0 | 85.7 | 80.4 | 72.9 | 4.28 |
| FLOWER | 99.4 | 95.8 | 90.7 | 84.9 | 77.8 | 4.53 |
| VLA-Adapter | 99.1 | 94.6 | 88.8 | 82.8 | 76.5 | 4.42 |
| Seer | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| **AVA-VLA** | **99.6** | **97.6** | **94.1** | **89.9** | **84.1** | **4.65** |

## Setup

Follow the setup instructions from [OpenVLA-OFT](https://github.com/moojink/openvla-oft). Compared with OpenVLA-OFT, the main AVA-VLA setup difference is **the AVA-VLA transformers fork** declared in `pyproject.toml`. If you need to install or override it explicitly, use:

```bash
pip install "transformers @ git+https://github.com/LiAuto-DSR/transformers-avavla.git"
```

## LIBERO Evaluation

Follow the LIBERO evaluation environment setup from [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

Download the LIBERO checkpoint from [LiAuto-DSR/avavla-libero-4in1](https://huggingface.co/LiAuto-DSR/avavla-libero-4in1), then run:

```bash
export DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export MUJOCO_EGL_DEVICE_ID=$DEVICE_ID
export EGL_DEVICE_ID=$DEVICE_ID
export TOKENIZERS_PARALLELISM=false

python ./experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint LiAuto-DSR/avavla-libero-4in1 \
  --local_log_dir ./logs/libero \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --multi_frame.temporal_strategy "attn_weight" \
  --multi_frame.temporal_feature_source "action" \
  --multi_frame.temporal_feature_layer -2 \
  --multi_frame.attn_weight_force_eager_attn True \
  --multi_frame.attn_weight_extra_layer_ids "(i for i in range(0, 32))" \
  --multi_frame.attn_weight_head_type "softmaxscore" \
  --multi_frame.attn_weight_score_config "(1.9, 0.1, 0.0)" \
  --multi_frame.attn_weight_sink_ids "[68,75,180,187,324,331,436,443]" \
  --multi_frame.attn_weight_sink_weight 1.0
```

Set `--task_suite_name` to `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10` to evaluate a specific suite.

## CALVIN Evaluation

Follow the CALVIN evaluation environment setup from [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter). Set `CALVIN_ROOT` to your local CALVIN checkout.

Download the CALVIN checkpoint from [LiAuto-DSR/avavla-calvin-abc2d](https://huggingface.co/LiAuto-DSR/avavla-calvin-abc2d), then run:

```bash
export CALVIN_ROOT=/path/to/calvin
export DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export NVIDIA_VISIBLE_DEVICES=$DEVICE_ID
export PYTHONPATH="$CALVIN_ROOT/calvin_models:$PYTHONPATH"
export PYOPENGL_PLATFORM=egl
export PYGLET_HEADLESS=1
export DISPLAY=""
export EGL_DEVICE_ID=$DEVICE_ID
export LIBGL_ALWAYS_SOFTWARE=0
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
export TOKENIZERS_PARALLELISM=false

python ./experiments/robot/calvin/run_calvin_eval.py \
  --pretrained_checkpoint LiAuto-DSR/avavla-calvin-abc2d \
  --local_log_dir ./logs/calvin \
  --unnorm_key "calvin_abc" \
  --multi_frame.temporal_strategy "attn_weight" \
  --multi_frame.temporal_feature_source "action" \
  --multi_frame.temporal_feature_layer -2 \
  --multi_frame.attn_weight_force_eager_attn True \
  --multi_frame.attn_weight_extra_layer_ids "(i for i in range(0, 32))" \
  --multi_frame.attn_weight_head_type "softmaxscore" \
  --multi_frame.attn_weight_score_config "(1.0, -1.0, 1.0)" \
  --multi_frame.attn_weight_sink_ids "[68,75,180,187,324,331,436,443]" \
  --multi_frame.attn_weight_sink_weight 1.0
```

## Citation

If you find AVA-VLA useful for your work, please cite:

```bibtex
@article{xiao2025ava,
  title={AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention},
  author={Xiao, Lei and Li, Jifeng and Gao, Juntao and Ye, Feiyang and Jin, Yan and Qian, Jingjing and Zhang, Jing and Wu, Yong and Yu, Xiaoyuan},
  journal={arXiv preprint arXiv:2511.18960},
  year={2025}
}
```

## Acknowledgements

AVA-VLA is built on [OpenVLA-OFT](https://github.com/moojink/openvla-oft). We thank [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [OpenVLA](https://github.com/openvla/openvla), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [CALVIN](https://github.com/mees/calvin), [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter), and the broader open-source robotics community for their code, models, benchmarks, and evaluation tools.

This repository is released under the Apache License 2.0. Portions derived from OpenVLA-OFT remain subject to the OpenVLA-OFT license notices included in this repository.

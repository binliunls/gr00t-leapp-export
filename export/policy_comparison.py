# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compare original/modified/exported policies using open-loop action chunk rollout.

Unlike policy_comparison.py, this script keeps every action from each predicted
chunk. It runs inference every action_horizon steps, appends the whole chunk to
the predicted trajectory, and compares the resulting rollout against ground truth.
"""

import argparse
import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import gr00t
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from leapp.inference_manager import InferenceManager

from policy_modifications import make_modifications
from utils import get_policy_and_dataset, set_all_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.7-3B")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(gr00t.__file__)), "demo_data/droid_sample"
        ),
    )
    parser.add_argument("--embodiment_tag", type=str, default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    parser.add_argument("--model_yaml_path", type=str, default="exported_gr00t/exported_gr00t.yaml")
    parser.add_argument("--max_steps", type=int, default=350)
    parser.add_argument("--traj_id", type=int, default=0)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--use_exported", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show_plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_action_chunk(value, action_horizon):
    """Return an action chunk as (horizon, dim), tolerating exported output shapes."""
    arr = _to_numpy(value)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        if arr.size % action_horizon != 0:
            raise ValueError(
                f"Cannot reshape flat action output of length {arr.size} "
                f"into horizon {action_horizon}"
            )
        arr = arr.reshape(action_horizon, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected action chunk with shape (horizon, dim), got {arr.shape}")
    return arr


def infer_decoded_action_horizon(policy, dataset, joint_names, traj_id, initial_noise):
    """Probe one policy call to get the actual decoded per-joint chunk length."""
    data = build_gr00t_input(dataset, policy, step_index=0, traj_id=traj_id)
    set_all_seeds(42)
    action, _ = policy.get_action(data, initial_noise=initial_noise)
    horizons = {
        joint_name: _as_action_chunk(action[joint_name], initial_noise.shape[1]).shape[0]
        for joint_name in joint_names
    }
    unique_horizons = set(horizons.values())
    if len(unique_horizons) != 1:
        raise ValueError(f"Decoded action horizons differ by joint: {horizons}")
    return unique_horizons.pop()


def _inference_steps(max_steps, action_horizon):
    return list(range(0, max_steps, action_horizon))


def build_gr00t_input(dataset, policy, step_index, traj_id):
    modality_config = policy.get_modality_config()
    step_data = extract_step_data(
        dataset[traj_id],
        step_index=step_index,
        modality_configs=modality_config,
        embodiment_tag=policy.embodiment_tag,
        allow_padding=False,
    )
    return {
        "video": {key: np.stack(value)[None] for key, value in step_data.images.items()},
        "state": {key: value[None] for key, value in step_data.states.items()},
        "action": {key: value[None] for key, value in step_data.actions.items()},
        "language": {
            modality_config["language"].modality_keys[0]: [[step_data.text]],
        },
    }


def collect_policy_rollout_outputs(
    policy,
    dataset,
    joint_names,
    max_steps,
    action_horizon,
    model_action_horizon,
    policy_name,
    traj_id=0,
    initial_noise_list=None,
):
    outputs = {joint_name: [] for joint_name in joint_names}
    inference_steps = _inference_steps(max_steps, action_horizon)

    for call_idx, step_count in enumerate(inference_steps):
        data = build_gr00t_input(dataset, policy, step_index=step_count, traj_id=traj_id)

        set_all_seeds(42)
        if initial_noise_list is not None:
            action, _ = policy.get_action(data, initial_noise=initial_noise_list[call_idx])
        else:
            action, _ = policy.get_action(data)

        steps_to_append = min(action_horizon, max_steps - step_count)
        for joint_name in joint_names:
            action_chunk = _as_action_chunk(action[joint_name], model_action_horizon)
            if action_chunk.shape[0] < steps_to_append:
                raise ValueError(
                    f"{policy_name} produced only {action_chunk.shape[0]} actions for "
                    f"'{joint_name}', but action_horizon={action_horizon} requires "
                    f"{steps_to_append}. Use --action_horizon <= {action_chunk.shape[0]}."
                )
            outputs[joint_name].extend(action_chunk[:steps_to_append])

        print(
            f"  {policy_name}: inference at step {step_count}; "
            f"rolled out {steps_to_append} actions ({len(outputs[joint_names[0]])}/{max_steps})"
        )

    for joint_name in joint_names:
        outputs[joint_name] = np.asarray(outputs[joint_name])[:max_steps]

    return outputs


def collect_exported_policy_rollout_outputs(
    exported_policy,
    dataset,
    modality_config,
    embodiment_tag,
    joint_names,
    video_keys,
    initial_noise_list,
    max_steps,
    action_horizon,
    model_action_horizon,
    traj_id=0,
):
    outputs = {joint_name: [] for joint_name in joint_names}
    episode_data = dataset[traj_id]

    for call_idx, step_count in enumerate(_inference_steps(max_steps, action_horizon)):
        step_data = extract_step_data(
            episode_data,
            step_index=step_count,
            modality_configs=modality_config,
            embodiment_tag=embodiment_tag,
            allow_padding=False,
        )

        inputs = {}
        for state_name, state_data in step_data.states.items():
            inputs[f"preprocess_state/{state_name}"] = torch.from_numpy(state_data).float()

        for video_key in video_keys:
            video_data = np.stack(step_data.images[video_key])
            inputs[f"preprocess_video/{video_key}"] = torch.from_numpy(video_data).to(torch.float32)

        inputs["action_head/initial_noise"] = initial_noise_list[call_idx]

        set_all_seeds(42)
        policy_outputs = exported_policy.run_policy(inputs)

        steps_to_append = min(action_horizon, max_steps - step_count)
        for joint_name in joint_names:
            output_key = f"decode_action/{joint_name}"
            if output_key not in policy_outputs:
                continue
            action_chunk = _as_action_chunk(policy_outputs[output_key], model_action_horizon)
            if action_chunk.shape[0] < steps_to_append:
                raise ValueError(
                    f"Exported policy produced only {action_chunk.shape[0]} actions for "
                    f"'{joint_name}', but action_horizon={action_horizon} requires "
                    f"{steps_to_append}. Use --action_horizon <= {action_chunk.shape[0]}."
                )
            outputs[joint_name].extend(action_chunk[:steps_to_append])

        print(
            f"  Exported: inference at step {step_count}; "
            f"rolled out {steps_to_append} actions ({len(outputs[joint_names[0]])}/{max_steps})"
        )

    for joint_name in joint_names:
        outputs[joint_name] = np.asarray(outputs[joint_name])[:max_steps] if outputs[joint_name] else None

    return outputs


def collect_ground_truth(episode_data, modality_config, embodiment_tag, joint_names, max_steps):
    gt_outputs = {joint_name: [] for joint_name in joint_names}

    for step_count in range(max_steps):
        step_data = extract_step_data(
            episode_data,
            step_index=step_count,
            modality_configs=modality_config,
            embodiment_tag=embodiment_tag,
            allow_padding=False,
        )
        for joint_name in joint_names:
            gt_outputs[joint_name].append(step_data.actions[joint_name][0])

    for joint_name in joint_names:
        gt_outputs[joint_name] = np.asarray(gt_outputs[joint_name])

    return gt_outputs


def compute_error_statistics(original_data, comparison_data):
    stats = []

    for dim in range(original_data.shape[1]):
        orig_dim = original_data[:, dim]
        comp_dim = comparison_data[:, dim]
        error_dim = np.abs(orig_dim - comp_dim)
        data_std = orig_dim.std() or 1.0
        rmse = np.sqrt(np.mean(error_dim**2))

        stats.append(
            {
                "dim": dim,
                "data_std": orig_dim.std(),
                "mean_error": error_dim.mean(),
                "max_error": error_dim.max(),
                "rmse": rmse,
                "nrmse": rmse / data_std,
                "p60": np.percentile(error_dim, 60),
                "p75": np.percentile(error_dim, 75),
                "p90": np.percentile(error_dim, 90),
                "p99": np.percentile(error_dim, 99),
                "p999": np.percentile(error_dim, 99.9),
            }
        )

    return stats


def print_error_statistics(joint_name, stats, max_steps):
    print(f"\n{'=' * 90}")
    print(f"{joint_name} ({len(stats)} dimensions, {max_steps} rolled-out steps)")
    print(f"{'=' * 90}")

    for stat in stats:
        print(f"\n  Dimension {stat['dim']}:")
        print(f"    Data std:      {stat['data_std']:.6f}")
        print(f"    Mean error:    {stat['mean_error']:.6e}")
        print(f"    Max error:     {stat['max_error']:.6e}")
        print(f"    RMSE:          {stat['rmse']:.6e}")
        print(f"    NRMSE:         {stat['nrmse']:.6e}  (RMSE / data_std)")
        print(
            f"    Percentiles:   p60={stat['p60']:.2e}, p75={stat['p75']:.2e}, "
            f"p90={stat['p90']:.2e}, p99={stat['p99']:.2e}, p99.9={stat['p999']:.2e}"
        )


def get_effective_max_steps(episode_data, modality_config, requested_max_steps):
    max_positive_delta = max(
        max(config.delta_indices) for config in modality_config.values() if config.delta_indices
    )
    available_steps = len(episode_data) - max_positive_delta
    if available_steps <= 0:
        raise ValueError(
            f"Episode length {len(episode_data)} is too short for max delta {max_positive_delta}"
        )

    effective_max_steps = min(requested_max_steps, available_steps)
    if effective_max_steps < requested_max_steps:
        print(
            f"Requested {requested_max_steps} steps, but only {effective_max_steps} are valid "
            f"for episode length {len(episode_data)} and max delta {max_positive_delta}."
        )
    return effective_max_steps


def plot_joint_comparison(
    joint_name,
    gt_data,
    original_data,
    comparison_data,
    exported,
    action_horizon,
    output_dir=".",
):
    os.makedirs(output_dir, exist_ok=True)
    num_dims = gt_data.shape[1]

    fig, axes = plt.subplots(nrows=num_dims, ncols=1, figsize=(12, 2.5 * num_dims))
    fig.suptitle(
        f"{joint_name} rollout comparison (inference every {action_horizon} steps)",
        fontsize=14,
        fontweight="bold",
    )

    if num_dims == 1:
        axes = [axes]

    comparison_label = "Exported Policy" if exported else "Modified Policy"
    inference_steps = _inference_steps(len(gt_data), action_horizon)
    for dim, ax in enumerate(axes):
        ax.plot(gt_data[:, dim], label="GT Action", color="green", alpha=0.7, linewidth=2)
        ax.plot(original_data[:, dim], label="Original Policy", color="blue", alpha=0.7, linewidth=1.5)
        ax.plot(
            comparison_data[:, dim],
            label=comparison_label,
            color="red",
            alpha=0.7,
            linestyle="--",
            linewidth=1.5,
        )

        for idx, step in enumerate(inference_steps):
            ax.plot(
                step,
                gt_data[step, dim],
                "ro",
                label="Inference point" if idx == 0 else None,
            )

        ax.set_title(f"Dimension {dim}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"policy_comparison_rollout_{joint_name}.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n  Saved: {output_path}")

    return fig


def plot_policy_comparison_rollout(
    max_steps=100,
    output_dir=".",
    show_plots=True,
    use_exported=True,
    exported_model_path="exported_gr00t/exported_gr00t.yaml",
    model_path="nvidia/GR00T-N1.7-3B",
    dataset_path=None,
    embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    video_backend="torchcodec",
    traj_id=0,
    action_horizon=None,
):
    print("Loading config info...")
    set_all_seeds(42)
    temp_policy, dataset = get_policy_and_dataset(model_path, dataset_path, embodiment_tag, video_backend)
    modality_config = temp_policy.get_modality_config()
    embodiment_tag = temp_policy.embodiment_tag
    model_action_horizon = temp_policy.model.action_head.config.action_horizon
    action_dim = temp_policy.model.action_head.action_dim
    device = temp_policy.model.device
    requested_action_horizon = action_horizon
    if requested_action_horizon is not None and requested_action_horizon > model_action_horizon:
        raise ValueError(
            f"Requested action_horizon={requested_action_horizon}, but model horizon is "
            f"{model_action_horizon}"
        )

    episode_data = dataset[traj_id]
    print(f"Episode length: {len(episode_data)}")
    max_steps = get_effective_max_steps(episode_data, modality_config, max_steps)

    first_step_data = extract_step_data(
        episode_data,
        step_index=0,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        allow_padding=False,
    )
    joint_names = list(first_step_data.actions.keys())
    video_keys = list(first_step_data.images.keys())
    probe_noise = torch.randn(1, model_action_horizon, action_dim, device=device, dtype=torch.float32)
    decoded_action_horizon = infer_decoded_action_horizon(
        temp_policy, dataset, joint_names, traj_id, probe_noise
    )
    if requested_action_horizon is None:
        action_horizon = decoded_action_horizon
    elif requested_action_horizon > decoded_action_horizon:
        raise ValueError(
            f"Requested action_horizon={requested_action_horizon}, but decoded policy outputs "
            f"only {decoded_action_horizon} actions per inference. Use --action_horizon "
            f"{decoded_action_horizon} or smaller."
        )
    else:
        action_horizon = requested_action_horizon
    print(f"Found joints: {joint_names}")
    print(f"Found video keys: {video_keys}")
    print(
        f"Rolling out {max_steps} steps with action_horizon={action_horizon} "
        f"(decoded chunk length: {decoded_action_horizon})"
    )

    del temp_policy
    gc.collect()
    torch.cuda.empty_cache()

    print("\nCollecting ground truth actions...")
    gt_outputs = collect_ground_truth(
        episode_data, modality_config, embodiment_tag, joint_names, max_steps
    )
    print("  Done collecting ground truth.")

    inference_steps = _inference_steps(max_steps, action_horizon)
    print("\nPre-generating initial noise for deterministic comparison...")
    set_all_seeds(42)
    initial_noise_list = [
        torch.randn(1, model_action_horizon, action_dim, device=device, dtype=torch.float32)
        for _ in inference_steps
    ]
    print(
        f"  Generated {len(initial_noise_list)} noise tensors of shape "
        f"(1, {model_action_horizon}, {action_dim})"
    )

    print("\nLoading original policy...")
    set_all_seeds(42)
    original_policy, dataset = get_policy_and_dataset(model_path, dataset_path, embodiment_tag, video_backend)
    time_start = time.time()
    original_outputs = collect_policy_rollout_outputs(
        original_policy,
        dataset,
        joint_names,
        max_steps,
        action_horizon,
        model_action_horizon,
        "Original",
        traj_id=traj_id,
        initial_noise_list=initial_noise_list,
    )
    original_time_taken = time.time() - time_start
    del original_policy
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed original policy memory.")

    comparison_outputs = None
    if not use_exported:
        print("\nLoading modified policy...")
        set_all_seeds(42)
        modified_policy, dataset = get_policy_and_dataset(
            model_path, dataset_path, embodiment_tag, video_backend
        )
        modified_policy = make_modifications(modified_policy)
        time_start = time.time()
        comparison_outputs = collect_policy_rollout_outputs(
            modified_policy,
            dataset,
            joint_names,
            max_steps,
            action_horizon,
            model_action_horizon,
            "Modified",
            traj_id=traj_id,
            initial_noise_list=initial_noise_list,
        )
        comparison_time_taken = time.time() - time_start
        del modified_policy, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print("  Freed modified policy memory.")
    else:
        print("\nLoading exported policy...")
        exported_policy = InferenceManager(exported_model_path)
        mock_inputs = exported_policy.get_mock_input()
        print("run warm up the policy")
        for _ in range(5):
            with torch.inference_mode():
                _ = exported_policy.run_policy(mock_inputs)
        print("  Warm up complete")

        time_start = time.time()
        comparison_outputs = collect_exported_policy_rollout_outputs(
            exported_policy,
            dataset,
            modality_config,
            embodiment_tag,
            joint_names,
            video_keys,
            initial_noise_list,
            max_steps,
            action_horizon,
            model_action_horizon,
            traj_id=traj_id,
        )
        comparison_time_taken = time.time() - time_start
        del exported_policy, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print("  Freed exported policy memory.")

    print("\n" + "=" * 90)
    print("Analysis Results (per dimension, across rolled-out steps)")
    print("=" * 90)

    for joint_name in joint_names:
        gt_data = gt_outputs[joint_name]
        original_data = original_outputs[joint_name]
        comparison_data = comparison_outputs[joint_name]
        if comparison_data is None:
            raise KeyError(f"No comparison output collected for joint '{joint_name}'")
        if original_data.shape != comparison_data.shape:
            raise ValueError(
                f"Shape mismatch for {joint_name}: original={original_data.shape}, "
                f"comparison={comparison_data.shape}"
            )
        if gt_data.shape != original_data.shape:
            raise ValueError(
                f"Shape mismatch for {joint_name}: gt={gt_data.shape}, "
                f"original={original_data.shape}. The rollout did not cover the full "
                "ground-truth window."
            )

        stats = compute_error_statistics(original_data, comparison_data)
        print_error_statistics(joint_name, stats, max_steps)

        plot_joint_comparison(
            joint_name,
            gt_data,
            original_data,
            comparison_data,
            exported=use_exported,
            action_horizon=action_horizon,
            output_dir=output_dir,
        )

    if show_plots:
        plt.show()

    print("  Original policy:")
    print(f"  time taken: {original_time_taken} seconds")
    print(f"  time per rollout step: {original_time_taken / max_steps} seconds")
    print("  Comparison policy:")
    print(f"  time taken: {comparison_time_taken} seconds")
    print(f"  time per rollout step: {comparison_time_taken / max_steps} seconds")
    print(f"  time per inference call: {comparison_time_taken / len(inference_steps)} seconds")


if __name__ == "__main__":
    args = parse_args()
    plot_policy_comparison_rollout(
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        use_exported=args.use_exported,
        exported_model_path=args.model_yaml_path,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        traj_id=args.traj_id,
        action_horizon=args.action_horizon,
    )

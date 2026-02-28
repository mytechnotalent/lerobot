#!/usr/bin/env python3
"""
Main entry point for the LeRobot Simulation Environment.

Demonstrates the full pipeline: environment instantiation, data collection
via teleoperation or random policy, behaviour-cloning training, evaluation,
and optional live visualization.  Run directly with ``python run_sim.py``
or import individual components for custom workflows.

Usage examples::

    # Full automated pipeline (collect, train, evaluate)
    python run_sim.py --task pusht --mode train

    # Keyboard teleoperation with live rendering
    python run_sim.py --task reach --mode teleop

    # Evaluate a previously trained policy
    python run_sim.py --task pick_place --mode eval
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import numpy as np

from lerobot_sim.datasets.dataset_manager import SimDatasetRecorder
from lerobot_sim.envs.configs import (
    PickPlaceSimConfig,
    PushTSimConfig,
    ReachSimConfig,
    SimEnvConfig,
)
from lerobot_sim.envs.factory import make_sim_env
from lerobot_sim.policies.policy import MLPPolicy, RandomPolicy
from lerobot_sim.policies.trainer import Trainer, TrainerConfig
from lerobot_sim.teleop.keyboard_teleop import KeyboardTeleop
from lerobot_sim.visualization.visualizer import SimVisualizer

# ======================================================================
# Configuration builders
# ======================================================================


def _build_env_config(task: str) -> SimEnvConfig:
    """Return the environment configuration for a given task name.

    Args:
        task: One of ``'pusht'``, ``'pick_place'``, or ``'reach'``.

    Returns:
        A concrete ``SimEnvConfig`` instance.

    Raises:
        ValueError: If the task name is not recognised.
    """
    registry = {
        "pusht": PushTSimConfig,
        "pick_place": PickPlaceSimConfig,
        "reach": ReachSimConfig,
    }
    if task not in registry:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(registry)}")
    return registry[task]()


def _build_policy(cfg: SimEnvConfig, hidden_dim: int = 256) -> MLPPolicy:
    """Create an MLP policy sized for the given environment configuration.

    Args:
        cfg: Environment configuration with ``state_dim`` and ``action_dim``.
        hidden_dim: Width of hidden layers.

    Returns:
        An ``MLPPolicy`` instance.
    """
    state_dim = getattr(cfg, "state_dim", 2)
    action_dim = getattr(cfg, "action_dim", 2)
    return MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)


def _build_trainer_config(args: argparse.Namespace) -> TrainerConfig:
    """Construct a ``TrainerConfig`` from parsed CLI arguments.

    Args:
        args: Namespace from ``argparse``.

    Returns:
        A ``TrainerConfig`` instance.
    """
    return TrainerConfig(
        num_demo_episodes=args.demo_episodes,
        num_train_steps=args.train_steps,
        num_eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
    )


# ======================================================================
# Mode runners
# ======================================================================


def _run_train(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Execute the full collect, train, evaluate pipeline and save best model.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    policy = _build_policy(env_cfg, hidden_dim=args.hidden_dim)
    trainer_cfg = _build_trainer_config(args)
    trainer = Trainer(env_cfg=env_cfg, policy=policy, config=trainer_cfg)
    results = trainer.run()
    print(f"\nFinal eval: {results['eval_metrics']}")
    print(f"Best model saved to: {args.output_dir}/best_policy.npz")


def _load_trained_policy(args: argparse.Namespace) -> MLPPolicy:
    """Load the best saved policy checkpoint from output_dir.

    Args:
        args: Parsed CLI arguments with ``output_dir``.

    Returns:
        An ``MLPPolicy`` with restored weights.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    import os

    best = os.path.join(args.output_dir, "best_policy.npz")
    if not os.path.exists(best):
        raise FileNotFoundError(f"No checkpoint at {best}. Run --mode train first.")
    print(f"Loading checkpoint: {best}")
    return MLPPolicy.load(best)


def _run_eval(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Load the best trained policy and run inference/evaluation.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    policy = _load_trained_policy(args)
    trainer_cfg = _build_trainer_config(args)
    trainer = Trainer(env_cfg=env_cfg, policy=policy, config=trainer_cfg)
    metrics = trainer.evaluate()
    print(f"\nInference results: {metrics}")


def _run_teleop_loop(
    env: Any, teleop: KeyboardTeleop, recorder: SimDatasetRecorder
) -> None:
    """Run the keyboard teleoperation loop for one episode.

    Args:
        env: Gymnasium environment instance.
        teleop: Keyboard teleop interface.
        recorder: Dataset recorder.
    """
    obs, _ = env.reset()
    recorder.start_episode()
    done = False
    while not done:
        action = teleop.get_action()
        recorder.record_step(obs, action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    recorder.end_episode()


def _run_teleop(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Run keyboard teleoperation to collect demonstrations.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    from lerobot_sim.envs.factory import _env_class_for_config

    env = _env_class_for_config(env_cfg)(env_cfg)
    action_dim = getattr(env_cfg, "action_dim", 2)
    teleop = KeyboardTeleop(action_dim=action_dim)
    recorder = SimDatasetRecorder(output_dir=args.output_dir)
    print("Teleop mode: use W/A/S/D to move, Q to quit.")
    print("Recording one episode via terminal input …")
    _run_teleop_loop(env, teleop, recorder)
    recorder.save_metadata()
    print(f"Episode recorded to {args.output_dir}")


def _run_visualize(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Run a random policy with live visualization.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    from lerobot_sim.envs.factory import _env_class_for_config

    env = _env_class_for_config(env_cfg)(env_cfg)
    action_dim = getattr(env_cfg, "action_dim", 2)
    policy = RandomPolicy(action_dim=action_dim, seed=args.seed)
    viz = SimVisualizer(
        width=env_cfg.observation_width,
        height=env_cfg.observation_height,
        fps=env_cfg.fps,
    )
    _run_visualize_loop(env, policy, viz, env_cfg.episode_length)
    viz.close()


def _run_visualize_loop(
    env: Any, policy: Any, viz: SimVisualizer, max_steps: int
) -> None:
    """Step through the env, rendering every frame to the visualizer.

    Args:
        env: Gymnasium environment.
        policy: Policy providing actions.
        viz: Visualizer instance.
        max_steps: Maximum steps before stopping.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    episode = 0
    for step in range(max_steps):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        image = obs.get("pixels", env.render())
        alive = viz.render_frame(
            image, step=step, reward=total_reward, success=info.get("is_success", False)
        )
        if not alive:
            break
        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} done (reward={total_reward:.2f})")
            obs, _ = env.reset()
            total_reward = 0.0


# ======================================================================
# CLI
# ======================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description="LeRobot Simulation Environment")
    parser.add_argument(
        "--task", choices=["pusht", "pick_place", "reach"], default="pusht"
    )
    parser.add_argument(
        "--mode", choices=["train", "eval", "teleop", "visualize"], default="train"
    )
    parser.add_argument("--demo-episodes", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dir", default="./sim_output")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ======================================================================
# Dispatch
# ======================================================================


# Mapping from mode name to runner function
_MODE_DISPATCH = {
    "train": _run_train,
    "eval": _run_eval,
    "teleop": _run_teleop,
    "visualize": _run_visualize,
}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Parse CLI arguments
    args = _parse_args()

    # Build environment configuration from the selected task
    env_cfg = _build_env_config(args.task)
    print(f"Task: {args.task} | Mode: {args.mode} | Seed: {args.seed}")
    print(f"Env config: {env_cfg.task}, fps={env_cfg.fps}, obs={env_cfg.obs_type}")
    print("-" * 60)

    # Dispatch to the selected mode runner
    runner = _MODE_DISPATCH[args.mode]
    runner(env_cfg, args)

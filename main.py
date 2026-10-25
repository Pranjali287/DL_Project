"""
Main CLI entrypoint for GRAIL-NAV:
Generalization-Aware Deep Reinforcement Learning for Autonomous Robot Navigation in Unseen Obstacle Environments.
"""

import argparse
from pathlib import Path
import sys

from config import DEFAULT_ENV_CONFIG, DEFAULT_TRAIN_CONFIG, MODELS_DIR, RESULTS_DIR
from evaluation.evaluate import evaluate_agent, run_full_comparative_evaluation
from training.train import train_agent
from visualization.plot_results import generate_all_plots
from visualization.render_environment import render_navigation_episode


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GRAIL-NAV: Generalization-Aware DQN Robot Navigation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "visualize", "report"],
        help="Execution mode: train, evaluate, visualize, or report"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="fixed",
        choices=["fixed", "diverse"],
        help="Training or evaluation scenario: fixed (single layout) or diverse (procedural layouts)"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for number of training episodes"
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=50,
        help="Number of episodes for evaluation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=9005,
        help="Random seed for single trajectory visualization"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.mode == "train":
        print(f"\n>>> Running Training Mode: Scenario = {args.scenario.upper()}")
        train_agent(
            scenario=args.scenario,
            episodes_override=args.episodes
        )

    elif args.mode == "evaluate":
        print(f"\n>>> Running Evaluation Mode: Scenario = {args.scenario.upper()}")
        model_path = MODELS_DIR / f"dqn_{args.scenario}.pt"
        if not model_path.exists():
            print(f"[-] Error: Checkpoint {model_path} does not exist. Please train the agent first using:")
            print(f"    python main.py --mode train --scenario {args.scenario}")
            sys.exit(1)

        print(f"\n[1/2] Evaluating {args.scenario.capitalize()} Agent on Familiar Layout...")
        metrics_fam, _ = evaluate_agent(
            model_path=model_path,
            test_type="familiar",
            num_episodes=args.test_episodes
        )
        print(f"  * Familiar Success Rate: {metrics_fam.success_rate_pct:.1f}%")
        print(f"  * Familiar Collision Rate: {metrics_fam.collision_rate_pct:.1f}%")
        print(f"  * Familiar Avg Reward: {metrics_fam.avg_reward:.2f}")

        print(f"\n[2/2] Evaluating {args.scenario.capitalize()} Agent on Held-Out Unseen Layouts...")
        metrics_unseen, _ = evaluate_agent(
            model_path=model_path,
            test_type="unseen",
            num_episodes=args.test_episodes
        )
        print(f"  * Unseen Success Rate: {metrics_unseen.success_rate_pct:.1f}%")
        print(f"  * Unseen Collision Rate: {metrics_unseen.collision_rate_pct:.1f}%")
        print(f"  * Unseen Avg Reward: {metrics_unseen.avg_reward:.2f}")
        print(f"  * Generalization Gap (Success Drop): {metrics_fam.success_rate_pct - metrics_unseen.success_rate_pct:.1f}%")

    elif args.mode == "visualize":
        print("\n>>> Running Trajectory Visualization Mode")
        # Check available models
        model_diverse = MODELS_DIR / "dqn_diverse.pt"
        model_fixed = MODELS_DIR / "dqn_fixed.pt"

        target_model = model_diverse if model_diverse.exists() else model_fixed
        if not target_model.exists():
            print("[-] Error: No trained model checkpoint found in models/. Train a model first.")
            sys.exit(1)

        label = "Diverse Agent on Unseen Map" if target_model == model_diverse else "Fixed Agent on Unseen Map"
        img_path, success, steps, r = render_navigation_episode(
            model_path=target_model,
            seed=args.seed,
            scenario_label=label
        )
        print(f"[+] Trajectory render generated: {img_path}")

    elif args.mode == "report":
        print("\n>>> Generating Full Comparative Report and Visualizations")
        # Run comparative evaluation if models exist
        model_fixed = MODELS_DIR / "dqn_fixed.pt"
        model_diverse = MODELS_DIR / "dqn_diverse.pt"

        if not (model_fixed.exists() and model_diverse.exists()):
            print("[-] Error: Both dqn_fixed.pt and dqn_diverse.pt are required to produce comparative report.")
            print("    Please train both scenarios first:")
            print("    python main.py --mode train --scenario fixed")
            print("    python main.py --mode train --scenario diverse")
            sys.exit(1)

        run_full_comparative_evaluation(num_episodes=args.test_episodes)
        plots = generate_all_plots()
        print(f"\n[+] Full evaluation report and {len(plots)} publication plots generated in {RESULTS_DIR}")


if __name__ == "__main__":
    main()

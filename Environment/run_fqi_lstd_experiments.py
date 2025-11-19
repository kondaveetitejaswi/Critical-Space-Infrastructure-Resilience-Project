import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Environment.original_toy_mdp import ToyConstellationMDP
from exact_solver import ExactDPSolver
from Environment.FQI_LSTD_implementation import FittedQIteration, LSTD_API
from Environment.FQI_LSTD_implementation import ADPEvaluator
import time


def create_visualizations(mdp, V_optimal, V_fqi, V_api, policy_optimal, policy_fqi, policy_api, fqi, api, dp_solver, results):
    '''
    Create comprehensive visualizations as per experimental protocol:
    1. Value function heatmaps
    2. Policy comparison grids
    3. Learning curves
    4. Runtime vs accuracy plots
    '''

    fig = plt.figure(figsize=(18, 12))
    
    # 1. Value Function Heatmaps (DP, FQI, API)
    ax1 = plt.subplot(3, 4, 1)
    create_value_heatmap(ax1, mdp, V_optimal, "V* (DP Baseline)")

    ax2 = plt.subplot(3, 4, 2)
    create_value_heatmap(ax2, mdp, V_fqi, "V_FQI")

    ax3 = plt.subplot(3, 4, 3)
    create_value_heatmap(ax3, mdp, V_api, "V_API")

    # 2. Value Function Error Heatmaps
    ax4 = plt.subplot(3, 4, 4)
    error_fqi = np.abs(V_fqi - V_optimal)
    create_value_heatmap(ax4, mdp, error_fqi, "|V_FQI - V*|")

    # 3. Policy Comparison Grids
    ax5 = plt.subplot(3, 4, 5)
    create_policy_comparison(ax5, mdp, policy_optimal, "π* (DP)")

    ax6 = plt.subplot(3, 4, 6)
    create_policy_comparison(ax6, mdp, policy_fqi, "π_FQI")

    ax7 = plt.subplot(3, 4, 7)
    create_policy_comparison(ax7, mdp, policy_api, "π_API")

    ax8 = plt.subplot(3, 4, 8)
    create_policy_match_grid(ax8, mdp, policy_optimal, policy_fqi, policy_api)

    # 4. Convergence Curves
    ax9 = plt.subplot(3, 4, 9)
    plot_convergence_curve(ax9, dp_solver, fqi, api, "DP")

    ax10 = plt.subplot(3, 4, 10)
    plot_convergence_curve(ax10, dp_solver, fqi, api, "FQI")

    ax11 = plt.subplot(3, 4, 11)
    plot_convergence_curve(ax11, dp_solver, fqi, api, "API")

    ax12 = plt.subplot(3, 4, 12)
    plot_runtime_vs_accuracy(ax12, results)

    plt.suptitle('ADP Comprehensive Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('ADP_complete_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nVisualization saved as: ADP_complete_analysis.png")


def create_value_heatmap(ax, mdp, V, title):
    '''
    Create value function heatmap for operational count vs spares
    '''
    value_matrix = np.zeros((len(mdp.op_counts), 2))

    for oc_idx, oc in enumerate(mdp.op_counts):
        for sp in [0, 1]:
            values_combo = []
            for h in mdp.allowed_health:
                for cov in [0, 1]:
                    state = (oc, sp, h, cov)
                    if state in mdp.state_to_idx:
                        values_combo.append(V[mdp.state_to_idx[state]])

            if values_combo:
                value_matrix[oc_idx, sp] = np.mean(values_combo)

    im = ax.imshow(value_matrix, cmap='viridis', aspect='auto')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Spare Status', fontsize=8)
    ax.set_ylabel('Operational Count', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"], fontsize=7)
    ax.set_yticks(range(len(mdp.op_counts)))
    ax.set_yticklabels(mdp.op_counts, fontsize=7)

    # Add value annotations
    for i in range(len(mdp.op_counts)):
        for j in range(2):
            ax.text(j, i, f'{value_matrix[i, j]:.1f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if value_matrix[i, j] < value_matrix.max() / 2 else 'black')
    plt.colorbar(im, ax=ax)


def create_policy_comparison(ax, mdp, policy, title):
    '''
    Create policy heatmap showing action distribution
    '''
    policy_matrix = np.full((len(mdp.op_counts), 2), -1)

    for oc_idx, oc in enumerate(mdp.op_counts):
        for sp in [0, 1]:
            state = (oc, sp, 1.0, 1)
            if state in mdp.state_to_idx:
                state_idx = mdp.state_to_idx[state]
                policy_matrix[oc_idx, sp] = policy[state_idx]

    im = ax.imshow(policy_matrix, cmap='tab10', aspect='auto', vmin=0, vmax=3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Spare Status', fontsize=8)
    ax.set_ylabel('Operational Count', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'], fontsize=7)
    ax.set_yticks(range(len(mdp.op_counts)))
    ax.set_yticklabels(mdp.op_counts, fontsize=7)

    # Add action labels
    action_names = ["NO", "REP", "ACT", "BST"]
    for i in range(len(mdp.op_counts)):
        for j in range(2):
            if policy_matrix[i, j] >= 0:
                ax.text(j, i, action_names[int(policy_matrix[i, j])],
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')


def create_policy_match_grid(ax, mdp, policy_dp, policy_fqi, policy_api):
    '''Show where FQI and API policies match DP baseline'''
    match_matrix = np.zeros((len(mdp.op_counts), 2))

    for oc_idx, oc in enumerate(mdp.op_counts):
        for sp in [0, 1]:
            matches = 0
            total = 0
            for h in mdp.allowed_health:
                for cov in [0, 1]:
                    state = (oc, sp, h, cov)
                    if state in mdp.state_to_idx:
                        idx = mdp.state_to_idx[state]
                        if policy_fqi[idx] == policy_dp[idx]:
                            matches += 0.5
                        if policy_api[idx] == policy_dp[idx]:
                            matches += 0.5

                        total += 1

            if total > 0:
                match_matrix[oc_idx, sp] = matches / total

    im = ax.imshow(match_matrix, cmap="RdYlGn", aspect='auto', vmin=0, vmax=1)
    ax.set_title('Policy Agreement\n(FQI + API vs DP)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Spare Status', fontsize=8)
    ax.set_ylabel('Operational Count', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"], fontsize=7)
    ax.set_yticks(range(len(mdp.op_counts)))
    ax.set_yticklabels(mdp.op_counts, fontsize=7)

    # Add percentage annotations
    for i in range(len(mdp.op_counts)):
        for j in range(2):
            ax.text(j, i, f'{match_matrix[i, j] * 100:.0f}%',
                    ha='center', va='center', fontsize=8,
                    color='white' if match_matrix[i, j] < 0.5 else 'black',
                    fontweight='bold')

    plt.colorbar(im, ax=ax, label='Match Ratio')


def plot_convergence_curve(ax, dp_solver, fqi, api, algo_type):
    """Plot convergence curves for each algorithm"""
    if algo_type == "DP":
        if hasattr(dp_solver, 'delta_history') and dp_solver.delta_history:
            iterations = range(len(dp_solver.delta_history))
            deltas = dp_solver.delta_history
            ax.plot(iterations, deltas, 'b-', linewidth=2, label="DP")
            ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax.set_xlabel('Iteration', fontsize=8)
            ax.set_ylabel('Delta', fontsize=8)
            ax.set_title('DP Convergence', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    elif algo_type == "FQI":
        if hasattr(fqi, 'convergence_deltas') and fqi.convergence_deltas:
            iterations = range(len(fqi.convergence_deltas))
            deltas = fqi.convergence_deltas
            ax.plot(iterations, deltas, 'g-', linewidth=2, label='FQI')
            ax.axhline(y=fqi.tol, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax.set_xlabel('Iteration', fontsize=8)
            ax.set_ylabel('ΔΘ (log scale)', fontsize=8)
            ax.set_title('FQI Convergence', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    elif algo_type == "API":
        if hasattr(api, 'convergence_deltas') and api.convergence_deltas:
            iterations = range(len(api.convergence_deltas))
            policy_changes = api.convergence_deltas
            ax.plot(iterations, policy_changes, color='orange', linewidth=2, label='API-LSTD')
            ax.set_xlabel('Iteration', fontsize=8)
            ax.set_ylabel('Policy Changes', fontsize=8)
            ax.set_title('API-LSTD Convergence', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)


def plot_runtime_vs_accuracy(ax, results):
    """Plot runtime vs accuracy trade-off"""
    algorithms = list(results.keys())
    runtimes = [results[algo]['time'] for algo in algorithms]
    match_ratios = [results[algo]['match_ratio'] * 100 for algo in algorithms]

    colors = {'DP': 'blue', 'FQI': 'green', 'API-LSTD': 'orange'}

    for i, algo in enumerate(algorithms):
        ax.scatter(runtimes[i], match_ratios[i],
                  s=200, c=colors.get(algo, 'gray'),
                  alpha=0.7, edgecolors='black', linewidth=2,
                  label=algo)
        ax.text(runtimes[i], match_ratios[i] - 3, algo,
               ha='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Runtime (seconds)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Policy Match Ratio (%)', fontsize=9, fontweight='bold')
    ax.set_title('Runtime vs Accuracy', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='lower right')

    # Add annotation for best trade-off
    if len(algorithms) > 1:
        non_dp_algos = [a for a in algorithms if a != 'DP']
        if non_dp_algos:
            best_idx = max(range(len(non_dp_algos)),
                          key=lambda i: match_ratios[algorithms.index(non_dp_algos[i])] /
                                       (runtimes[algorithms.index(non_dp_algos[i])] + 0.001))
            best_algo = non_dp_algos[best_idx]
            ax.annotate('Best Trade-off',
                       xy=(results[best_algo]['time'], results[best_algo]['match_ratio'] * 100),
                       xytext=(10, -15), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=7, fontweight='bold')


def run_full_experiment():
    """
    Execute complete experimental protocol:
    1. Compute DP baseline
    2. Run FQI
    3. Run API-LSTD
    4. Evaluate and compare
    5. Visualize results
    """

    # Parameters from experimental protocol
    gamma = 0.95
    lambda_reg = 1e-3
    max_iter = 1000
    tol = 1e-4
    n_eval_episodes = 1000

    print("\n" + "=" * 70)
    print("ADP EXPERIMENTAL PROTOCOL - SATELLITE CONSTELLATION MDP")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Discount factor (γ): {gamma}")
    print(f"  Regularization (λ): {lambda_reg}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Tolerance: {tol}")
    print(f"  Evaluation episodes: {n_eval_episodes}")
    print("=" * 70)

    # ========================================================================
    # STEP 1: COMPUTE DP BASELINE
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: COMPUTING DYNAMIC PROGRAMMING BASELINE")
    print("=" * 70)

    mdp = ToyConstellationMDP()
    dp_solver = ExactDPSolver(mdp, gamma=gamma)

    start_time = time.time()
    V_optimal, policy_optimal = dp_solver.value_iteration(theta=1e-6)
    dp_time = time.time() - start_time

    print(f"DP completed in {dp_time:.4f} seconds")
    print(f"DP iterations: {len(dp_solver.delta_history)}")
    print(f"Mean optimal value: {np.mean(V_optimal):.4f}")

    # ========================================================================
    # STEP 2: RUN FITTED Q-ITERATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FITTED Q-ITERATION")
    print("=" * 70)

    fqi = FittedQIteration(mdp, gamma=gamma, lambda_reg=lambda_reg,
                          max_iters=max_iter, tol=tol)

    start_time = time.time()
    theta_fqi, policy_fqi = fqi.solve()
    fqi_time = time.time() - start_time
    V_fqi = fqi.get_value_function()

    print(f"FQI completed in {fqi_time:.4f} seconds")
    print(f"FQI iterations: {len(fqi.convergence_deltas)}")
    print(f"Mean FQI value: {np.mean(V_fqi):.4f}")

    # ========================================================================
    # STEP 3: RUN API-LSTD
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: APPROXIMATE POLICY ITERATION (LSTD)")
    print("=" * 70)

    api = LSTD_API(mdp, gamma=gamma, lambda_reg=lambda_reg,
                   max_iters=max_iter, tol=tol)

    start_time = time.time()
    w_api, policy_api = api.solve()
    api_time = time.time() - start_time
    V_api = api.get_value_function()

    print(f"API-LSTD completed in {api_time:.4f} seconds")
    print(f"API-LSTD iterations: {len(api.convergence_deltas)}")
    print(f"Mean API value: {np.mean(V_api):.4f}")

    # ========================================================================
    # STEP 4: POLICY EVALUATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: POLICY EVALUATION")
    print("=" * 70)

    evaluator = ADPEvaluator(mdp, V_optimal, policy_optimal, gamma=gamma)

    # Initialize results dictionary
    results = {}

    # DP (baseline)
    print("\nEvaluating DP policy with Monte Carlo rollouts...")
    mc_return_dp, mc_se_dp = evaluator.monte_carlo_evaluation(
        policy_optimal, n_episodes=n_eval_episodes, seed=42
    )
    results['DP'] = {
        'rmse': 0.0,
        'match_ratio': 1.0,
        'mc_return': (mc_return_dp, mc_se_dp),
        'iterations': len(dp_solver.delta_history),
        'time': dp_time
    }
    print(f"  DP MC Return: {mc_return_dp:.4f} ± {mc_se_dp:.4f}")

    # FQI
    print("\nEvaluating FQI policy...")
    rmse_fqi = evaluator.compute_rmse(V_fqi)
    match_fqi = evaluator.compute_policy_match_ratio(policy_fqi)
    print(f"  FQI RMSE: {rmse_fqi:.4f}")
    print(f"  FQI Policy Match: {match_fqi * 100:.2f}%")

    print("  Running Monte Carlo rollouts for FQI...")
    mc_return_fqi, mc_se_fqi = evaluator.monte_carlo_evaluation(
        policy_fqi, n_episodes=n_eval_episodes, seed=42
    )
    print(f"  FQI MC Return: {mc_return_fqi:.4f} ± {mc_se_fqi:.4f}")

    results['FQI'] = {
        'rmse': rmse_fqi,
        'match_ratio': match_fqi,
        'mc_return': (mc_return_fqi, mc_se_fqi),
        'iterations': len(fqi.convergence_deltas),
        'time': fqi_time
    }

    # API-LSTD
    print("\nEvaluating API-LSTD policy...")
    rmse_api = evaluator.compute_rmse(V_api)
    match_api = evaluator.compute_policy_match_ratio(policy_api)
    print(f"  API-LSTD RMSE: {rmse_api:.4f}")
    print(f"  API-LSTD Policy Match: {match_api * 100:.2f}%")

    print("  Running Monte Carlo rollouts for API-LSTD...")
    mc_return_api, mc_se_api = evaluator.monte_carlo_evaluation(
        policy_api, n_episodes=n_eval_episodes, seed=42
    )
    print(f"  API-LSTD MC Return: {mc_return_api:.4f} ± {mc_se_api:.4f}")

    results['API-LSTD'] = {
        'rmse': rmse_api,
        'match_ratio': match_api,
        'mc_return': (mc_return_api, mc_se_api),
        'iterations': len(api.convergence_deltas),
        'time': api_time
    }

    # Generate comprehensive report
    evaluator.generate_comparison_report(results)

    # ========================================================================
    # STEP 5: VISUALIZATION AND DIAGNOSTICS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 70)

    create_visualizations(mdp, V_optimal, V_fqi, V_api,
                         policy_optimal, policy_fqi, policy_api,
                         fqi, api, dp_solver, results)

    # ========================================================================
    # ADDITIONAL ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("ADDITIONAL ANALYSIS")
    print("=" * 70)

    # Compute relative performance
    print("\nRelative Performance (compared to DP baseline):")
    print("-" * 70)

    dp_return = results['DP']['mc_return'][0]

    for algo in ['FQI', 'API-LSTD']:
        algo_return = results[algo]['mc_return'][0]
        relative_perf = (algo_return / dp_return) * 100
        speedup = dp_time / results[algo]['time']

        print(f"\n{algo}:")
        print(f"  Performance: {relative_perf:.2f}% of DP optimal")
        print(f"  Speedup: {speedup:.2f}x faster than DP")
        print(f"  Policy agreement: {results[algo]['match_ratio'] * 100:.2f}%")
        print(f"  Value function RMSE: {results[algo]['rmse']:.4f}")

    # State-level analysis
    print("\n" + "=" * 70)
    print("STATE-LEVEL POLICY DISAGREEMENTS")
    print("=" * 70)

    print("\nStates where FQI disagrees with DP:")
    disagreements_fqi = 0
    for state_idx, state in enumerate(mdp.states):
        if policy_fqi[state_idx] != policy_optimal[state_idx]:
            disagreements_fqi += 1
            if disagreements_fqi <= 5:  # Show first 5
                print(f"  State {state}: DP={mdp.actions[policy_optimal[state_idx]]}, "
                      f"FQI={mdp.actions[policy_fqi[state_idx]]}")
    if disagreements_fqi > 5:
        print(f"  ... and {disagreements_fqi - 5} more states")

    print("\nStates where API-LSTD disagrees with DP:")
    disagreements_api = 0
    for state_idx, state in enumerate(mdp.states):
        if policy_api[state_idx] != policy_optimal[state_idx]:
            disagreements_api += 1
            if disagreements_api <= 5:  # Show first 5
                print(f"  State {state}: DP={mdp.actions[policy_optimal[state_idx]]}, "
                      f"API={mdp.actions[policy_api[state_idx]]}")
    if disagreements_api > 5:
        print(f"  ... and {disagreements_api - 5} more states")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 70)

    print("\n✓ All algorithms completed successfully")
    print(f"✓ DP baseline established: V* computed in {dp_time:.4f}s")
    print(f"✓ FQI converged in {len(fqi.convergence_deltas)} iterations ({fqi_time:.4f}s)")
    print(f"✓ API-LSTD converged in {len(api.convergence_deltas)} iterations ({api_time:.4f}s)")
    print(f"✓ Monte Carlo evaluation: {n_eval_episodes} episodes per algorithm")
    print(f"✓ Comprehensive visualizations generated")

    print("\nKey Findings:")
    print(f"  - FQI achieved {results['FQI']['match_ratio'] * 100:.1f}% policy agreement with DP")
    print(f"  - API-LSTD achieved {results['API-LSTD']['match_ratio'] * 100:.1f}% policy agreement with DP")
    print(f"  - FQI value RMSE: {results['FQI']['rmse']:.4f}")
    print(f"  - API-LSTD value RMSE: {results['API-LSTD']['rmse']:.4f}")

    # Determine which ADP method performed better
    if results['FQI']['match_ratio'] > results['API-LSTD']['match_ratio']:
        best_adp = 'FQI'
    else:
        best_adp = 'API-LSTD'

    print(f"\n  → {best_adp} showed better overall approximation quality")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70 + "\n")

    return results, mdp, dp_solver, fqi, api


if __name__ == "__main__":
    # Run the complete experimental protocol
    results, mdp, dp_solver, fqi, api = run_full_experiment()

    # Optional: Save results to file
    import json

    # Convert results to JSON-serializable format
    results_serializable = {}
    for algo, data in results.items():
        results_serializable[algo] = {
            'rmse': float(data['rmse']),
            'match_ratio': float(data['match_ratio']),
            'mc_return_mean': float(data['mc_return'][0]),
            'mc_return_stderr': float(data['mc_return'][1]),
            'iterations': int(data['iterations']),
            'time': float(data['time'])
        }

    with open('adp_experiment_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("\nResults saved to: adp_experiment_results.json")
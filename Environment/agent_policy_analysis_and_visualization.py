import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\tejas\\Masters Thesis\\Critical-Space-Infrastructure-Resilience-Project\\resilience_output\\2.0_main_resilience_log.csv')

total = len(df)

action_prob = df['action'].value_counts(normalize=True)
print("Action Probabilities:")
print(action_prob)

attack_prob = df[df['under_attack'] == 1]['action'].value_counts(normalize=True)
no_attack_prob = df[df['under_attack'] == 0]['action'].value_counts(normalize=True)

comparison = pd.DataFrame({
    'Attack': attack_prob,
    'No Attack': no_attack_prob
}).fillna(0)

comparison['Percent Change (%)'] = (
    (comparison['Attack'] - comparison['No Attack']) /
    comparison['No Attack']
) * 100

print("\nAction Probability Comparison:\n")
print(comparison)

phase_probs = df.groupby('phase')['action'].value_counts(normalize=True).unstack().fillna(0)

print("\nAction Probabilities by Phase:\n")
print(phase_probs)

df['next_action'] = df['action'].shift(-1)

# Rows where attack happened
attack_rows = df[df['under_attack'] == 1]

# Check what happens next
reaction = attack_rows['next_action'].value_counts(normalize=True)

print("\nAgent reaction after attack:\n")
print(reaction)

# --- Conditional NO_OP rate ---

no_op_attack = (df[(df['under_attack'] == 1) & (df['action'] == 'NO_OP')].shape[0] /
                df[df['under_attack'] == 1].shape[0])

no_op_no_attack = (df[(df['under_attack'] == 0) & (df['action'] == 'NO_OP')].shape[0] /
                   df[df['under_attack'] == 0].shape[0])

# attack_counts = df[df['under_attack'] == 1]['action'].value_counts()
# no_attack_counts = df[df['under_attack'] == 0]['action'].value_counts()

print("\n--- Conditional NO_OP Rate ---")
print(f"P(NO_OP | Attack)     = {no_op_attack:.3f}")
print(f"P(NO_OP | No Attack)  = {no_op_no_attack:.3f}")

# print("\n--- Conditional NO_OP Rate ---")
# print(f"P(NO_OP | Attack)     = {attack_counts}")
# print(f"P(NO_OP | No Attack)  = {no_attack_counts}")

# # Specifically for NO_OP
# no_op_attack_count = df[(df['under_attack'] == 1) & (df['action'] == 'NO_OP')].shape[0]
# no_op_no_attack_count = df[(df['under_attack'] == 0) & (df['action'] == 'NO_OP')].shape[0]

# print("\n--- NO_OP COUNTS ---")
# print(f"NO_OP under attack     = {no_op_attack_count}")
# print(f"NO_OP without attack  = {no_op_no_attack_count}")

# Reaction delay Analysis

delays = []

for i in range(len(df) - 1):
    if df.loc[i, 'under_attack'] == 1:
        delay = 0
        for j in range(i+1, min(i+10, len(df))):
            if df.loc[j, 'action'] != 'NO_OP':
                delay = j - i
                break
        if delay > 0:
            delays.append(delay)

if delays:
    avg_delay = sum(delays) / len(delays)
    print(f"\nAverage Reaction Delay: {avg_delay:.2f} steps")
else:
    print("\nNo reactions observed after attacks.")

# Performance Impact After Attack

coverage_drops = []

for i in range(len(df) - 1):
    if df.loc[i, 'under_attack'] == 1:
        before = df.loc[i, 'coverage_pct']
        after = df.loc[i+1:i+5, 'coverage_pct'].min()
        drop = before - after
        coverage_drops.append(drop)

if coverage_drops:
    avg_drops = sum(coverage_drops) / len(coverage_drops)
    print(f"\nAverage Coverage Drop After Attack: {avg_drops:.2f}%")

# --- Reaction Effectiveness ---

improvements = []

for i in range(len(df) - 2):
    if df.loc[i, 'under_attack'] == 1:
        before = df.loc[i, 'coverage_pct']
        after = df.loc[i+2, 'coverage_pct']
        improvements.append(after - before)

if improvements:
    avg_improve = sum(improvements) / len(improvements)
    print("\n--- Reaction Effectiveness ---")
    print(f"Average recovery after attack: {avg_improve:.2f}%")
import pandas as pd

df = pd.read_csv('C:\\Users\\tejas\\Masters Thesis\\Critical-Space-Infrastructure-Resilience-Project\\resilience_output\\2.0_main_resilience_log.csv')


# Analysis fo the results that we got

phase_action = pd.crosstab(
    df['phase'],
    df['action'],
    normalize='index'
)

print("\nAction Distribution by Threat Phase:\n")
print(phase_action)

attack_action = pd.crosstab(
    df['under_attack'],
    df['action'],
    normalize='index'
)

print("\nAction Distribution by Attack Condition:\n")
print(attack_action)

p_activate_attack = attack_action.loc[1, 'ACTIVATE_SPARE']
p_activate_no_attack = attack_action.loc[0, 'ACTIVATE_SPARE']

print("\nP(ACTIVATE_SPARE | attack=1):", p_activate_attack)
print("P(ACTIVATE_SPARE | attack=0):", p_activate_no_attack)
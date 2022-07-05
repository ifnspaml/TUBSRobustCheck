from collections import namedtuple

# Attacks dictionary structure
Attack = namedtuple('Attack', ['classpath', 'classname', 'universal'])

# Define dictionary of attacks
ATTACKS = {
    'fgsm': Attack('robustness.attacks.fgsm', 'FGSM', False),
    'gd_uap': Attack('robustness.attacks.gd_uap', 'GDUAP', True),
    'i_fgsm': Attack('robustness.attacks.iterative_fgsm', 'IterativeFGSM', False),
    'metzen': Attack('robustness.attacks.metzen', 'Metzen', False),
    'metzen_uap': Attack('robustness.attacks.metzen_uap', 'MetzenUAP', True),
    'mi_fgsm': Attack('robustness.attacks.momentum_iterative_fgsm', 'MomentumIterativeFGSM', False),
    'pgd': Attack('robustness.attacks.pgd', 'PGD', False),
}

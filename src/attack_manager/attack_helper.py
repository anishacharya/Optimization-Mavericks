# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License
from .attack_models import (DriftAttack, Additive, Random,
                            BitFlipAttack, RandomSignFlipAttack)
from typing import Dict


def get_attack(attack_config: Dict):
    if attack_config["attack_model"] == 'drift':
        return DriftAttack(attack_config=attack_config)
    elif attack_config["attack_model"] == 'additive':
        return Additive(attack_config=attack_config)
    elif attack_config["attack_model"] == 'random':
        return Random(attack_config=attack_config)
    elif attack_config["attack_model"] == 'bit_flip':
        return BitFlipAttack(attack_config=attack_config)
    elif attack_config["attack_model"] == 'random_sign_flip':
        return RandomSignFlipAttack(attack_config=attack_config)
    else:
        return None

"""Greedy search + random character substitution (not a one-shot random typo baseline)."""

from textattack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapRandomCharacterSubstitution

from text_attack.medgemma_attack_common import char_level_constraints, maybe_set_seed


def attack(model_wrapper):
    maybe_set_seed()
    transformation = WordSwapRandomCharacterSubstitution(random_one=True)
    goal_function = UntargetedClassification(model_wrapper)
    search_method = GreedyWordSwapWIR()
    return Attack(
        goal_function,
        char_level_constraints(),
        transformation,
        search_method,
    )

from textattack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

from text_attack.medgemma_attack_common import char_level_constraints, maybe_set_seed


def attack(model_wrapper):
    maybe_set_seed()
    transformation = CompositeTransformation(
        [
            WordSwapNeighboringCharacterSwap(),
            WordSwapRandomCharacterSubstitution(),
            WordSwapRandomCharacterDeletion(),
            WordSwapRandomCharacterInsertion(),
        ]
    )

    goal_function = UntargetedClassification(model_wrapper)
    search_method = GreedyWordSwapWIR()
    return Attack(goal_function, char_level_constraints(), transformation, search_method)

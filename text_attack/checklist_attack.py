from textattack import Attack
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapChangeNumber,
    WordSwapContract,
    WordSwapExtend,
)

from text_attack.vlm_attack_common import (
    VLM_FROZEN_COLUMNS,
    VLM_INPUT_COLUMNS,
    maybe_set_seed,
)


def attack(model_wrapper):
    maybe_set_seed()
    transformation = CompositeTransformation(
        [
            WordSwapExtend(),
            WordSwapContract(),
            WordSwapChangeName(),
            WordSwapChangeNumber(),
            WordSwapChangeLocation(),
        ]
    )

    constraints = [
        RepeatModification(),
        InputColumnModification(VLM_INPUT_COLUMNS, VLM_FROZEN_COLUMNS),
    ]

    goal_function = UntargetedClassification(model_wrapper)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, transformation, search_method)

from textattack import Attack
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from text_attack.medgemma_attack_common import (
    VLM_FROZEN_COLUMNS,
    VLM_INPUT_COLUMNS,
    force_tensorflow_cpu,
    maybe_set_seed,
)


def attack(model_wrapper):
    maybe_set_seed()
    force_tensorflow_cpu()
    transformation = CompositeTransformation(
        [
            WordSwapRandomCharacterInsertion(
                random_one=True,
                letters_to_insert=" ",
                skip_first_char=True,
                skip_last_char=True,
            ),
            WordSwapRandomCharacterDeletion(
                random_one=True, skip_first_char=True, skip_last_char=True
            ),
            WordSwapNeighboringCharacterSwap(
                random_one=True, skip_first_char=True, skip_last_char=True
            ),
            WordSwapHomoglyphSwap(),
            WordSwapEmbedding(max_candidates=5),
        ]
    )

    constraints = [
        RepeatModification(),
        StopwordModification(),
        InputColumnModification(VLM_INPUT_COLUMNS, VLM_FROZEN_COLUMNS),
        UniversalSentenceEncoder(threshold=0.8),
    ]

    goal_function = UntargetedClassification(model_wrapper)
    search_method = GreedyWordSwapWIR(wir_method="delete")
    return Attack(goal_function, constraints, transformation, search_method)

"""TextAttack-native random character substitution search.

This samples independent random character substitutions from the original prompt
until the TextAttack query budget is exhausted or the prediction flips.
"""

from __future__ import annotations

import os
import random
import string
from pathlib import Path

from textattack import Attack
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import SearchMethod
from textattack.transformations import Transformation

from text_attack.medgemma_attack_common import (
    char_level_constraints,
    maybe_set_seed,
    register_for_parallel_pickling,
)


class RandomCharacterSubstitution(Transformation):
    def __init__(self, *, char_rate: float = 0.05, num_chars: int | None = None):
        self.char_rate = char_rate
        self.num_chars = num_chars
        self._attempt = 0

    def _get_transformations(self, current_text, indices_to_modify):
        eligible_positions = []
        for word_idx in sorted(indices_to_modify):
            word = current_text.words[word_idx]
            for char_idx, char in enumerate(word):
                if char.isalnum():
                    eligible_positions.append((word_idx, char_idx, char))

        if not eligible_positions:
            return []

        num_chars = self.num_chars
        if num_chars is None:
            num_chars = max(1, int(round(len(eligible_positions) * self.char_rate)))
        num_chars = min(int(num_chars), len(eligible_positions))

        # Deterministic per attack attempt, while still sampling independently.
        rng = random.Random(self._attempt)
        self._attempt += 1

        new_words = {}
        for word_idx, char_idx, old_char in rng.sample(eligible_positions, num_chars):
            word_chars = list(new_words.get(word_idx, current_text.words[word_idx]))
            pool = [c for c in string.ascii_letters if c.lower() != old_char.lower()]
            if not pool:
                continue
            new_char = rng.choice(pool)
            if old_char.isupper():
                new_char = new_char.upper()
            elif old_char.islower():
                new_char = new_char.lower()
            word_chars[char_idx] = new_char
            new_words[word_idx] = "".join(word_chars)

        if not new_words:
            return []

        return [
            current_text.replace_words_at_indices(
                list(new_words.keys()),
                list(new_words.values()),
            )
        ]

    @property
    def deterministic(self):
        return False

    def extra_repr_keys(self):
        return ["char_rate", "num_chars"]


class IndependentRandomSearch(SearchMethod):
    def perform_search(self, initial_result):
        best_result = initial_result
        search_over = False

        while not search_over:
            transformed_texts = self.get_transformations(
                initial_result.attacked_text,
                original_text=initial_result.attacked_text,
            )
            if not transformed_texts:
                break

            results, search_over = self.get_goal_results(transformed_texts)
            if not results:
                break

            candidate = results[0]
            if candidate.score > best_result.score:
                best_result = candidate
            if candidate.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                return candidate

        return best_result

    @property
    def is_black_box(self):
        return True


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def attack(model_wrapper):
    maybe_set_seed()
    transformation = RandomCharacterSubstitution(
        char_rate=float(os.getenv("RANDOM_CHAR_RATE", "0.05")),
        num_chars=_optional_int_env("RANDOM_CHAR_NUM_CHARS"),
    )
    goal_function = UntargetedClassification(model_wrapper)
    search_method = IndependentRandomSearch()
    return Attack(
        goal_function,
        char_level_constraints(),
        transformation,
        search_method,
    )


register_for_parallel_pickling(
    "text_attack.medgemma_random_char_search_attack",
    Path(__file__),
    {
        "RandomCharacterSubstitution": RandomCharacterSubstitution,
        "IndependentRandomSearch": IndependentRandomSearch,
        "attack": attack,
    },
)

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _reward(scores: Dict[str, Any]) -> float:
    return float(scores.get("reward", scores.get("margin", 0.0)))


def _chosen_pred(scores: Dict[str, Any], prediction_source: str) -> str:
    return str(scores.get(prediction_source) or scores.get("pred") or "")


def _is_attack(scores: Dict[str, Any], truth_label: str, prediction_source: str) -> bool:
    chosen = _chosen_pred(scores, prediction_source)
    return bool(truth_label) and bool(chosen) and chosen.lower() != str(truth_label).lower()


def _history_entry(
    *,
    step: int,
    question: str,
    scores: Dict[str, Any],
    parent_scores: Dict[str, Any],
    transition: Tuple[str, str],
    prediction_source: str,
    truth_label: str,
) -> Dict[str, Any]:
    truth_new = float(scores.get("truth_score", 0.0))
    truth_prev = float(parent_scores.get("truth_score", 0.0))
    margin_new = float(scores.get("margin", 0.0))
    margin_prev = float(parent_scores.get("margin", 0.0))
    reward_new = _reward(scores)
    reward_prev = _reward(parent_scores)
    return {
        "step": step,
        "question": question,
        "reward": round(reward_new, 3),
        "delta_reward": round(reward_new - reward_prev, 3),
        "reward_source": scores.get("reward_source"),
        "truth_score": round(truth_new, 3),
        "best_other_score": round(float(scores.get("best_other_score", 0.0)), 3),
        "delta_truth_score": round(truth_new - truth_prev, 3),
        "attack_margin": round(margin_new, 3),
        "delta_attack_margin": round(margin_new - margin_prev, 3),
        "transition": f"{transition[0]} : {transition[1]}",
        "pred": scores.get("pred"),
        "real_pred": scores.get("real_pred"),
        "chosen_pred": _chosen_pred(scores, prediction_source),
        "prediction_source": prediction_source,
        "truth": truth_label,
    }


@dataclass
class _SearchState:
    question: str
    scores: Dict[str, Any]
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0


class GeneticSearch:
    """
    Genetic-style iteratively accepts minimal edits
    that reduce the truth score or flip the prediction.
    """

    def __init__(
        self,
        scorer,
        proposer,
        apply_edit,
        truth_label: str,
        prediction_source: str = "pred",
        max_steps: int = 50,
        generations_per_step: int = 6,
        attempt_multiplier: int = 40,
        max_evaluations: Optional[int] = None,
    ):
        self.scorer = scorer
        self.proposer = proposer
        self.apply_edit = apply_edit
        self.truth_label = truth_label
        self.prediction_source = prediction_source
        self.max_steps = max_steps
        self.generations_per_step = generations_per_step
        self.attempt_multiplier = attempt_multiplier
        self.max_evaluations = max_evaluations
        self.visited_questions = set()

    def search(
        self,
        initial_question: str,
        initial_scores: Dict[str, float],
        progress=None,
    ) -> Dict[str, Any]:
        question = initial_question
        current_scores = dict(initial_scores)
        reward_prev = float(current_scores.get("reward", current_scores.get("margin", 0.0)))
        margin_prev = float(current_scores.get("margin", 0.0))
        truth_prev = float(current_scores.get("truth_score", 0.0))
        transitions: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []

        max_steps = max(1, int(self.max_steps))
        per_step_max_generations = max(1, int(self.generations_per_step))
        max_attempts = max_steps * max(1, int(self.attempt_multiplier))
        max_proposer_calls = (
            self.max_evaluations
            if self.max_evaluations is not None
            else max_attempts * per_step_max_generations
        )

        attempts = 0
        accepted_steps = 0
        evaluations = 0
        proposer_calls = 0
        budget_exhausted = False
        miscls = False
        self.visited_questions.add(question)

        while accepted_steps < max_steps and not budget_exhausted:
            attempts += 1

            per_step_generations = 0
            best_choice = None
            while per_step_generations < per_step_max_generations:
                per_step_generations += 1
                proposer_calls += 1
                if proposer_calls > max_proposer_calls:
                    budget_exhausted = True
                    break
                pairs = self.proposer(question, transitions) or []
                if not pairs:
                    if attempts >= max_attempts:
                        break
                    continue

                for prev, new in pairs:
                    if self.max_evaluations is not None and evaluations >= self.max_evaluations:
                        budget_exhausted = True
                        break
                    mutated_question = self.apply_edit(question, prev, new)
                    if mutated_question == question or mutated_question in self.visited_questions:
                        continue
                    try:
                        scores = self.scorer(mutated_question)
                    except Exception:
                        continue
                    evaluations += 1
                    self.visited_questions.add(mutated_question)
                    reward_new = float(scores.get("reward", scores.get("margin", 0.0)))
                    delta_reward = reward_new - reward_prev
                    delta_truth = scores["truth_score"] - truth_prev
                    improves_reward = delta_reward > 1e-9
                    chosen_pred = str(
                        scores.get(self.prediction_source) or scores.get("pred") or ""
                    )
                    flips = bool(chosen_pred) and chosen_pred.lower() != self.truth_label.lower()
                    if flips:
                        best_choice = (mutated_question, (prev, new), scores)
                        break
                    if not improves_reward:
                        continue

                    if best_choice is None:
                        best_choice = (mutated_question, (prev, new), scores)
                    else:
                        _, _, best_scores = best_choice
                        better = False
                        best_reward = float(
                            best_scores.get("reward", best_scores.get("margin", 0.0))
                        )
                        if reward_new > best_reward + 1e-9:
                            better = True
                        elif abs(reward_new - best_reward) <= 1e-9:
                            if scores["truth_score"] < best_scores["truth_score"] - 1e-9:
                                better = True
                        if better:
                            best_choice = (mutated_question, (prev, new), scores)

                if best_choice is not None or budget_exhausted:
                    break

                if attempts >= max_attempts:
                    break

            if best_choice is None:
                if budget_exhausted or attempts >= max_attempts:
                    break
                continue

            accepted_steps += 1
            if progress is not None:
                try:
                    progress.update(1)
                except Exception:
                    pass

            mutated_question, chosen, new_scores = best_choice

            pred = new_scores["pred"]
            chosen_pred = str(
                new_scores.get(self.prediction_source) or new_scores.get("pred") or ""
            )
            reward_new = float(new_scores.get("reward", new_scores.get("margin", 0.0)))
            margin_new = new_scores["margin"]
            truth_new = new_scores["truth_score"]
            delta_reward = reward_new - reward_prev
            delta_margin = margin_new - margin_prev
            delta_truth = truth_new - truth_prev
            best_other_score = float(new_scores.get("best_other_score", float("nan")))

            history.append(
                {
                    "step": accepted_steps,
                    "question": mutated_question,
                    "reward": round(reward_new, 3),
                    "delta_reward": round(delta_reward, 3),
                    "reward_source": new_scores.get("reward_source"),
                    "truth_score": round(truth_new, 3),
                    "best_other_score": round(best_other_score, 3),
                    "delta_truth_score": round(delta_truth, 3),
                    "attack_margin": round(margin_new, 3),
                    "delta_attack_margin": round(delta_margin, 3),
                    "transition": (f"{chosen[0]} : {chosen[1]}" if chosen else None),
                    "pred": pred,
                    "real_pred": new_scores.get("real_pred"),
                    "chosen_pred": chosen_pred,
                    "prediction_source": self.prediction_source,
                    "truth": self.truth_label,
                }
            )

            if chosen is not None:
                transitions.append(
                    {"prev": chosen[0], "new": chosen[1], "delta": float(delta_truth)}
                )

            question = mutated_question
            current_scores = new_scores
            reward_prev = reward_new
            margin_prev = margin_new
            truth_prev = truth_new

            if chosen_pred.lower() != self.truth_label.lower():
                miscls = True
                break

        return {
            "success": miscls,
            "question": question,
            "scores": current_scores,
            "transitions": transitions,
            "history": history,
            "evaluations": evaluations,
        }


class RandomSearch:
    """Randomly expands edit candidates under a fixed evaluation budget."""

    def __init__(
        self,
        scorer,
        proposer,
        apply_edit,
        truth_label: str,
        prediction_source: str = "pred",
        max_iterations: int = 80,
        max_depth: int = 8,
        max_children_per_expand: int = 3,
    ):
        self.scorer = scorer
        self.proposer = proposer
        self.apply_edit = apply_edit
        self.truth_label = truth_label
        self.prediction_source = prediction_source
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.max_children = max_children_per_expand
        self.visited_questions = set()

    def search(
        self,
        initial_question: str,
        initial_scores: Dict[str, Any],
        progress=None,
    ) -> Dict[str, Any]:
        root = _SearchState(initial_question, dict(initial_scores))
        frontier: List[_SearchState] = [root]
        self.visited_questions.add(initial_question)
        best = root
        history: List[Dict[str, Any]] = []
        evaluations = 0
        attack_state: Optional[_SearchState] = (
            root if _is_attack(root.scores, self.truth_label, self.prediction_source) else None
        )

        while evaluations < self.max_iterations and attack_state is None:
            expandable = [state for state in frontier if state.depth < self.max_depth]
            if not expandable:
                break
            parent = random.choice(expandable)
            pairs = list(self.proposer(parent.question, parent.transitions) or [])
            random.shuffle(pairs)

            expanded = 0
            for prev, new in pairs:
                if expanded >= self.max_children or evaluations >= self.max_iterations:
                    break
                mutated = self.apply_edit(parent.question, prev, new)
                if mutated == parent.question or mutated in self.visited_questions:
                    continue
                try:
                    scores = self.scorer(mutated)
                except Exception:
                    continue

                evaluations += 1
                expanded += 1
                if progress is not None:
                    try:
                        progress.update(1)
                    except Exception:
                        pass

                delta_truth = float(scores.get("truth_score", 0.0)) - float(
                    parent.scores.get("truth_score", 0.0)
                )
                transitions = parent.transitions + [
                    {"prev": prev, "new": new, "delta": delta_truth}
                ]
                child = _SearchState(mutated, scores, transitions, parent.depth + 1)
                frontier.append(child)
                self.visited_questions.add(mutated)
                history.append(
                    _history_entry(
                        step=evaluations,
                        question=mutated,
                        scores=scores,
                        parent_scores=parent.scores,
                        transition=(prev, new),
                        prediction_source=self.prediction_source,
                        truth_label=self.truth_label,
                    )
                )

                if _reward(scores) > _reward(best.scores):
                    best = child
                if _is_attack(scores, self.truth_label, self.prediction_source):
                    attack_state = child
                    break

            if expanded == 0:
                frontier.remove(parent)

        final_state = attack_state or best
        return {
            "success": attack_state is not None,
            "question": final_state.question,
            "scores": final_state.scores,
            "transitions": final_state.transitions,
            "history": history,
            "evaluations": evaluations,
        }


class GreedySearch:
    """At each step, evaluates local edit candidates and keeps the best reward."""

    def __init__(
        self,
        scorer,
        proposer,
        apply_edit,
        truth_label: str,
        prediction_source: str = "pred",
        max_iterations: int = 80,
        max_depth: int = 8,
        max_children_per_expand: int = 3,
    ):
        self.scorer = scorer
        self.proposer = proposer
        self.apply_edit = apply_edit
        self.truth_label = truth_label
        self.prediction_source = prediction_source
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.max_children = max_children_per_expand
        self.visited_questions = set()

    def search(
        self,
        initial_question: str,
        initial_scores: Dict[str, Any],
        progress=None,
    ) -> Dict[str, Any]:
        question = initial_question
        current_scores = dict(initial_scores)
        transitions: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        best_question = question
        best_scores = current_scores
        best_transitions: List[Dict[str, Any]] = []
        evaluations = 0
        depth = 0
        self.visited_questions.add(question)

        if _is_attack(current_scores, self.truth_label, self.prediction_source):
            return {
                "success": True,
                "question": question,
                "scores": current_scores,
                "transitions": transitions,
                "history": history,
                "evaluations": evaluations,
            }

        success = False
        while evaluations < self.max_iterations and depth < self.max_depth:
            parent_scores = current_scores
            pairs = list(self.proposer(question, transitions) or [])
            candidates = []
            for prev, new in pairs:
                if len(candidates) >= self.max_children or evaluations >= self.max_iterations:
                    break
                mutated = self.apply_edit(question, prev, new)
                if mutated == question or mutated in self.visited_questions:
                    continue
                try:
                    scores = self.scorer(mutated)
                except Exception:
                    continue

                evaluations += 1
                self.visited_questions.add(mutated)
                if progress is not None:
                    try:
                        progress.update(1)
                    except Exception:
                        pass
                candidates.append((mutated, (prev, new), scores))

                history.append(
                    _history_entry(
                        step=evaluations,
                        question=mutated,
                        scores=scores,
                        parent_scores=parent_scores,
                        transition=(prev, new),
                        prediction_source=self.prediction_source,
                        truth_label=self.truth_label,
                    )
                )

                if _reward(scores) > _reward(best_scores):
                    best_question = mutated
                    best_scores = scores
                    delta_truth = float(scores.get("truth_score", 0.0)) - float(
                        parent_scores.get("truth_score", 0.0)
                    )
                    best_transitions = transitions + [
                        {"prev": prev, "new": new, "delta": delta_truth}
                    ]
                if _is_attack(scores, self.truth_label, self.prediction_source):
                    best_question = mutated
                    best_scores = scores
                    delta_truth = float(scores.get("truth_score", 0.0)) - float(
                        parent_scores.get("truth_score", 0.0)
                    )
                    best_transitions = transitions + [
                        {"prev": prev, "new": new, "delta": delta_truth}
                    ]
                    success = True
                    break

            if success or not candidates:
                break

            question, transition, current_scores = max(
                candidates,
                key=lambda item: (_reward(item[2]), -float(item[2].get("truth_score", 0.0))),
            )
            delta_truth = float(current_scores.get("truth_score", 0.0)) - float(
                parent_scores.get("truth_score", 0.0)
            )
            transitions.append({"prev": transition[0], "new": transition[1], "delta": delta_truth})
            depth += 1

        final_question = best_question
        final_scores = best_scores
        final_transitions = best_transitions
        if not success and _reward(current_scores) >= _reward(best_scores):
            final_question = question
            final_scores = current_scores
            final_transitions = transitions

        return {
            "success": success,
            "question": final_question,
            "scores": final_scores,
            "transitions": final_transitions,
            "history": history,
            "evaluations": evaluations,
        }


class BeamSearch:
    """Layer-wise beam search over edit candidates under a fixed evaluation budget."""

    def __init__(
        self,
        scorer,
        proposer,
        apply_edit,
        truth_label: str,
        prediction_source: str = "pred",
        max_iterations: int = 80,
        max_depth: int = 8,
        max_children_per_expand: int = 3,
        beam_width: Optional[int] = None,
    ):
        self.scorer = scorer
        self.proposer = proposer
        self.apply_edit = apply_edit
        self.truth_label = truth_label
        self.prediction_source = prediction_source
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.max_children = max_children_per_expand
        self.beam_width = beam_width or max_children_per_expand
        self.visited_questions = set()

    def search(
        self,
        initial_question: str,
        initial_scores: Dict[str, Any],
        progress=None,
    ) -> Dict[str, Any]:
        root = _SearchState(initial_question, dict(initial_scores))
        self.visited_questions.add(initial_question)
        best = root
        history: List[Dict[str, Any]] = []
        evaluations = 0
        attack_state: Optional[_SearchState] = (
            root if _is_attack(root.scores, self.truth_label, self.prediction_source) else None
        )

        pool: List[_SearchState] = [root]

        while evaluations < self.max_iterations and attack_state is None:
            expandable = [state for state in pool if state.depth < self.max_depth]
            if not expandable:
                break
            expandable.sort(
                key=lambda state: (
                    _reward(state.scores),
                    -float(state.scores.get("truth_score", 0.0)),
                ),
                reverse=True,
            )
            beam = expandable[: max(1, int(self.beam_width))]
            evals_before = evaluations
            for parent in beam:
                if evaluations >= self.max_iterations:
                    break
                pairs = list(self.proposer(parent.question, parent.transitions) or [])
                expanded = 0
                for prev, new in pairs:
                    if expanded >= self.max_children or evaluations >= self.max_iterations:
                        break
                    mutated = self.apply_edit(parent.question, prev, new)
                    if mutated == parent.question or mutated in self.visited_questions:
                        continue
                    try:
                        scores = self.scorer(mutated)
                    except Exception:
                        continue

                    evaluations += 1
                    expanded += 1
                    self.visited_questions.add(mutated)
                    if progress is not None:
                        try:
                            progress.update(1)
                        except Exception:
                            pass

                    delta_truth = float(scores.get("truth_score", 0.0)) - float(
                        parent.scores.get("truth_score", 0.0)
                    )
                    child = _SearchState(
                        question=mutated,
                        scores=scores,
                        transitions=parent.transitions
                        + [{"prev": prev, "new": new, "delta": delta_truth}],
                        depth=parent.depth + 1,
                    )
                    pool.append(child)
                    history.append(
                        _history_entry(
                            step=evaluations,
                            question=mutated,
                            scores=scores,
                            parent_scores=parent.scores,
                            transition=(prev, new),
                            prediction_source=self.prediction_source,
                            truth_label=self.truth_label,
                        )
                    )

                    if _reward(scores) > _reward(best.scores):
                        best = child
                    if _is_attack(scores, self.truth_label, self.prediction_source):
                        attack_state = child
                        break
                if attack_state is not None:
                    break

            if evaluations == evals_before:
                break

        final_state = attack_state or best
        return {
            "success": attack_state is not None,
            "question": final_state.question,
            "scores": final_state.scores,
            "transitions": final_state.transitions,
            "history": history,
            "evaluations": evaluations,
        }

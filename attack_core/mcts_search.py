import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class MCTSNode:
    question: str
    scores: Dict[str, float]
    truth_label: str
    transition: Optional[Tuple[str, str]] = None
    parent: Optional["MCTSNode"] = None
    depth: int = 0
    delta_truth: float = 0.0
    visits: int = 0
    value: float = 0.0
    fully_expanded: bool = False  # proposer can no longer add new children here
    exhausted: bool = False  # this node and its whole subtree are explored
    children: List["MCTSNode"] = field(default_factory=list)

    @property
    def margin(self) -> float:
        return float(self.scores.get("margin", 0.0))

    @property
    def reward(self) -> float:
        return float(self.scores.get("reward", self.scores.get("margin", 0.0)))

    @property
    def pred(self) -> str:
        return str(self.scores.get("pred", ""))

    @property
    def real_pred(self) -> str:
        return str(self.scores.get("real_pred", "") or "")

    def chosen_pred(self, prediction_source: str) -> str:
        if prediction_source == "real_pred":
            return self.real_pred
        return self.pred

    def is_attack(self, prediction_source: str) -> bool:
        chosen = self.chosen_pred(prediction_source)
        return (
            bool(self.truth_label)
            and bool(chosen)
            and chosen.lower() != str(self.truth_label).lower()
        )


class MCTS:
    def __init__(
        self,
        scorer: Callable[[str], Dict[str, float]],
        proposer: Callable[[str, List[Dict[str, Any]]], List[Tuple[str, str]]],
        apply_edit: Callable[[str, str, str], str],
        truth_label: str,
        prediction_source: str = "pred",
        max_depth: int = 8,
        exploration: float = 1.4,
        max_iterations: int = 160,
        max_children_per_expand: int = 6,
    ):
        self.scorer = scorer
        self.proposer = proposer
        self.apply_edit = apply_edit
        self.truth_label = truth_label
        self.prediction_source = prediction_source
        self.max_depth = max_depth
        self.exploration = exploration
        self.max_iterations = max_iterations
        self.max_children = max_children_per_expand
        self.visited_questions = set()
        self.trace: List[Dict[str, Any]] = []

    def search(self, root_question: str, root_scores: Dict[str, float], progress=None):
        root = MCTSNode(
            question=root_question, scores=root_scores, truth_label=self.truth_label, depth=0
        )
        self.visited_questions.add(root_question)
        best_node = root
        attack_node = root if root.is_attack(self.prediction_source) else None

        evaluations = 0

        while evaluations < self.max_iterations and attack_node is None:
            node = self._select(root)
            if node.is_attack(self.prediction_source):
                attack_node = node
                break

            # _select returns either an expandable node or an exhausted one.
            if node.exhausted:
                if node is root:
                    break
                self._backprop(node, self._node_reward(node))
                continue

            new_children = self._expand(node)
            evaluations += len(new_children)
            if progress is not None and new_children:
                progress.update(len(new_children))
            if not new_children:
                # No new children available; keep this node's existing children
                # reachable but stop trying to grow it.
                node.fully_expanded = True
                self._backprop(node, self._node_reward(node))
                continue

            for child in new_children:
                self._backprop(child, self._node_reward(child))
                if child.reward > best_node.reward:
                    best_node = child
                if child.is_attack(self.prediction_source):
                    attack_node = child
                    break

        return {
            "root": root,
            "attack_node": attack_node,
            "best_node": best_node,
            "trace": self.trace,
            # Minimal tree snapshot: each node keeps only its score.
            "score_tree": self._score_tree(root),
        }

    def _can_expand(self, node: MCTSNode) -> bool:
        return (
            not node.fully_expanded
            and node.depth < self.max_depth
            and len(node.children) < self.max_children
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while True:
            if current.visits == 0 or self._can_expand(current):
                return current
            candidates = [ch for ch in current.children if not ch.exhausted]
            if not candidates:
                current.exhausted = True
                return current
            current = max(candidates, key=lambda ch: self._uct(current, ch))

    def _uct(self, parent: MCTSNode, child: MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + self.exploration * math.sqrt(
            math.log(parent.visits + 1) / child.visits
        )

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        transitions = self._path_transitions(node)
        pairs = self.proposer(node.question, transitions) or []
        # Avoid retrying the same edit pairs for a node.
        existing_pairs = {
            (str(ch.transition[0]).lower(), str(ch.transition[1]).lower())
            for ch in node.children
            if ch.transition
        }
        children = []
        for prev, new in pairs:
            mutated = self.apply_edit(node.question, prev, new)
            if mutated == node.question:
                continue
            if mutated in self.visited_questions:
                continue
            key_pair = (str(prev).lower(), str(new).lower())
            if key_pair in existing_pairs:
                continue
            try:
                scores = self.scorer(mutated)
            except Exception:
                continue

            child = MCTSNode(
                question=mutated,
                scores=scores,
                truth_label=self.truth_label,
                transition=(prev, new),
                parent=node,
                depth=node.depth + 1,
                delta_truth=scores.get("truth_score", 0.0) - node.scores.get("truth_score", 0.0),
            )
            node.children.append(child)
            self.visited_questions.add(mutated)
            existing_pairs.add(key_pair)
            children.append(child)
            self.trace.append(self._trace_entry(child))
            if len(children) >= self.max_children:
                break
        return children

    def _trace_entry(self, node: MCTSNode) -> Dict[str, Any]:
        parent = node.parent
        parent_margin = parent.scores.get("margin", 0.0) if parent else 0.0
        delta_margin = node.scores.get("margin", 0.0) - parent_margin
        parent_reward = parent.scores.get("reward", parent_margin) if parent else 0.0
        delta_reward = node.scores.get("reward", node.scores.get("margin", 0.0)) - parent_reward
        return {
            "depth": node.depth,
            "question": node.question,
            "transition": node.transition,
            "scores": dict(node.scores.get("scores", {})),
            "pred_gap": float(node.scores.get("pred_gap", node.scores.get("gap", 0.0))),
            "truth_score": float(node.scores.get("truth_score", 0.0)),
            "best_other_score": float(node.scores.get("best_other_score", 0.0)),
            "margin": float(node.scores.get("margin", 0.0)),
            "reward": float(node.scores.get("reward", node.scores.get("margin", 0.0))),
            "reward_source": node.scores.get("reward_source"),
            "delta_truth_score": float(node.delta_truth),
            "delta_margin": float(delta_margin),
            "delta_reward": float(delta_reward),
            "pred": node.pred,
            "real_pred": node.real_pred,
            "real_runner_up": node.scores.get("real_runner_up"),
            "real_runner_up_source": node.scores.get("real_runner_up_source"),
            "real_pred_sequence_score": node.scores.get("real_pred_sequence_score"),
            "real_runner_up_sequence_score": node.scores.get("real_runner_up_sequence_score"),
            "real_runner_up_score_fallback": node.scores.get("real_runner_up_score_fallback"),
            "generation_option_pred": node.scores.get("generation_option_pred"),
            "generation_option_pred_score": node.scores.get("generation_option_pred_score"),
            "generation_option_runner_up": node.scores.get("generation_option_runner_up"),
            "generation_option_runner_up_score": node.scores.get(
                "generation_option_runner_up_score"
            ),
            "generation_option_gap": node.scores.get("generation_option_gap"),
            "generation_option_margin": node.scores.get("generation_option_margin"),
            "chosen_pred": node.chosen_pred(self.prediction_source),
            "prediction_source": self.prediction_source,
            "decoded_output": node.scores.get("decoded_output"),
            "truth": node.truth_label,
            "real_label": node.truth_label,
        }

    def _path_transitions(self, node: MCTSNode) -> List[Dict[str, Any]]:
        seq = []
        cur = node
        while cur and cur.transition:
            seq.append(
                {"prev": cur.transition[0], "new": cur.transition[1], "delta": cur.delta_truth}
            )
            cur = cur.parent
        seq.reverse()
        return seq

    def _node_reward(self, node: MCTSNode) -> float:
        return float(node.scores.get("reward", node.scores.get("margin", 0.0)))

    def _backprop(self, node: MCTSNode, reward: float):
        cur = node
        while cur:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    def _score_tree(self, node: MCTSNode) -> Dict[str, Any]:
        """Return a nested dict of the tree where nodes store only their score."""

        def _walk(cur: MCTSNode) -> Dict[str, Any]:
            return {
                "score": float(cur.reward),
                "children": [_walk(ch) for ch in cur.children],
            }

        return _walk(node)

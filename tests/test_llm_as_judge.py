import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from evaluation.llm_as_judge import judge_generated_questions as judge


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.rubric_path = Path(judge.__file__).with_name("categories.json")
        self.rubric, self.allowed, self.categories = judge.load_rubric(
            self.rubric_path
        )

    def test_extracts_json_from_wrapped_response(self):
        parsed, end = judge.extract_json_object('result: {"confidence": 0.8} done')
        self.assertEqual(parsed, {"confidence": 0.8})
        self.assertIsInstance(end, int)

    def test_normalization_uses_allowed_values(self):
        result = judge.normalize_judgment(
            {
                "grammar_label": "grammatical",
                "weirdness_label": "none",
                "task_intent_label": "preserved",
                "context_fit_label": "fits",
                "preserves_output_constraints": True,
                "primary_category": "clean paraphrase",
                "confidence": 2,
            },
            self.allowed,
            self.categories,
            "What is shown?",
            "What does this image show?",
        )
        self.assertTrue(result["is_natural"])
        self.assertEqual(result["primary_category"], "clean_paraphrase")
        self.assertEqual(result["confidence"], 1.0)

    @patch.object(judge, "generate_judgment")
    def test_process_csv_writes_row_and_metric_outputs(self, generate):
        generate.return_value = (
            {
                "is_natural": True,
                "is_grammatical": True,
                "has_weird_artifacts": False,
                "preserves_task_intent": True,
                "preserves_output_constraints": True,
                "grammar_label": "correct",
                "weirdness_label": "none",
                "task_intent_label": "preserved",
                "context_fit_label": "fits",
                "primary_category": "clean_paraphrase",
                "confidence": 0.9,
                "reason_short": "Clear and faithful.",
            },
            '{"primary_category":"clean_paraphrase"}',
            True,
            100,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "model-a__model-b__mcts.csv"
            pd.DataFrame(
                [
                    {
                        "original_question": "What is shown?",
                        "final_question": "What does this image show?",
                    }
                ]
            ).to_csv(csv_path, index=False)

            stats = judge.process_csv(
                csv_path=csv_path,
                output_dir=root,
                rubric=self.rubric,
                allowed_labels=self.allowed,
                primary_categories=self.categories,
                judge_model_name="test/model",
                api_base="https://example.invalid",
                api_key="in-memory-test-value",
                site_url=None,
                app_name=None,
                max_new_tokens=100,
                temperature=0.0,
                timeout_seconds=1,
                max_retries=0,
                skip_existing=False,
                limit=None,
            )

            self.assertEqual(stats["successful_attack_count"], 1)
            self.assertEqual(stats["is_natural_rate"], 1.0)
            self.assertEqual(stats["has_weird_artifacts_rate"], 0.0)
            self.assertTrue(
                (root / "model-a__model-b__mcts.judge.csv").is_file()
            )
            metric_path = (
                root / "model-a__model-b__mcts.judge.statistics.json"
            )
            self.assertEqual(
                json.loads(metric_path.read_text())["parse_success_rate"], 1.0
            )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.similarity import calculate_sims


class SimilarityTests(unittest.TestCase):
    def test_identical_text_scores(self):
        text = "Is the lesion benign or malignant?"
        self.assertEqual(calculate_sims.levenshtein_distance(text, text), 0)
        self.assertAlmostEqual(calculate_sims.calculate_bleu(text, text), 100.0)
        self.assertAlmostEqual(calculate_sims.calculate_chrf(text, text), 100.0)
        rouge = calculate_sims.calculate_rouge(text, text)
        self.assertAlmostEqual(rouge["rouge1_fmeasure"], 1.0)
        self.assertAlmostEqual(rouge["rougeL_fmeasure"], 1.0)

    def test_native_attack_summary_schema_writes_all_base_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "task.attack_summary.csv"
            output_dir = root / "metrics"
            output_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "original_question": "Is the lesion benign or malignant?",
                        "final_question": "Classify the lesion as benign or malignant.",
                        "attack_success": True,
                    },
                    {
                        "original_question": "Is the lesion benign or malignant?",
                        "final_question": "Unsuccessful mutation.",
                        "attack_success": False,
                    }
                ]
            ).to_csv(input_path, index=False)

            stats = calculate_sims.process_csv(
                csv_path=input_path,
                output_dir=output_dir,
                include_semantic=False,
                emb_model=None,
                emb_tokenizer=None,
                include_perplexity=False,
                lm_model=None,
                lm_tokenizer=None,
                embedding_model_name=None,
                perplexity_model_name=None,
                skip_existing=False,
            )

            metrics = pd.read_csv(output_dir / "task.attack_summary.sims.csv")
            expected = {
                "levenshtein_distance",
                "bleu_score",
                "chrf_score",
                "rouge1_fmeasure",
                "rougeL_fmeasure",
                "base_question",
                "final_question",
            }
            self.assertTrue(expected.issubset(metrics.columns))
            self.assertEqual(stats["successful_attack_count"], 1)
            self.assertTrue(
                (output_dir / "task.attack_summary.statistics.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()

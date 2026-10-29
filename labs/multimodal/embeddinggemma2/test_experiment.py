"""Metric checks with synthetic vectors, never model performance results."""
import unittest
import numpy as np
from experiment import alignment, normalized, retrieval, local_path

class AlignmentTests(unittest.TestCase):
    def test_perfect_and_scrambled_pairs(self):
        x = np.eye(8, dtype=np.float32)
        rows = [dict(id=str(i), child_id=str(i // 2), session_id='s') for i in range(8)]
        perfect, pairs = alignment(x, x, rows)
        self.assertEqual(perfect['text_to_audio']['mrr'], 1)
        self.assertEqual(perfect['matched_minus_mismatched'], 1)
        self.assertEqual(pairs[0]['gap'], 0)
        scrambled, _ = alignment(x, np.roll(x, 1, axis=0), rows)
        self.assertEqual(scrambled['text_to_audio']['recall_at_1'], 0)
        self.assertLess(scrambled['matched_minus_mismatched'], 0)

    def test_asymmetric_retrieval_and_ties(self):
        scores = np.array([[.9, .8], [.95, .7]])
        self.assertEqual(retrieval(scores)['recall_at_1'], .5)
        self.assertEqual(retrieval(scores.T)['recall_at_1'], 0)
        self.assertEqual(retrieval(np.ones((3, 3)))['mrr'], 1 / 3)

    def test_invalid_vectors_and_paths(self):
        for x in (np.zeros((2, 3)), [[np.nan]], [1, 2]):
            with self.assertRaises(ValueError):
                normalized(x)
        with self.assertRaises(ValueError):
            local_path('../escape')

if __name__ == '__main__':
    unittest.main()

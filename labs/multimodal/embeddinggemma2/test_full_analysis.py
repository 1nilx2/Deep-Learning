import unittest
import numpy as np
from evaluate_exp1_full import ranks, grouped, per_query, uncertainty, weak_topic

class FullAnalysisTests(unittest.TestCase):
    def test_groups_keep_pair_order(self):
        scores=np.eye(6)
        groups=grouped(scores,['b','a','b','a','c','c'])
        self.assertEqual(groups['a']['n_candidates'],2)
        self.assertEqual(groups['a']['text_to_audio']['recall_at_1'],1)
        np.testing.assert_array_equal(ranks(np.ones((3,3))),[3,3,3])

    def test_child_bootstrap_and_pair_shuffle(self):
        rows=[dict(id=str(i),text='read a book',child_id=f'c{i//2}',session_id=f's{i//2}') for i in range(6)]
        scores=np.eye(6)
        per=per_query(scores,rows)
        self.assertTrue(all(p['text_rank']==1 for p in per))
        children,result=uncertainty(scores,rows,per,bootstrap=20,permutations=19)
        self.assertEqual(len(children),3)
        self.assertEqual(result['matched_minus_same_session_negative']['child_bootstrap_95ci'],[1.,1.])
        self.assertGreater(result['within_session_permutation']['one_sided_p'],0)
        self.assertEqual(weak_topic('read a book'),'book_story')
        self.assertEqual(weak_topic('a book and a truck'),'mixed')
        self.assertEqual(weak_topic('right here'),'unassigned')

if __name__=='__main__':
    unittest.main()

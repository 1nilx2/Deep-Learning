import unittest
from prepare_dialogues import windows, spoken_text
from experiment import require_dialogue_verification


def turn(start, end, raw='hello .', speaker='MOT', corrupted=False):
    return dict(span=(start,end), raw=raw, speaker=speaker, corrupted=corrupted, line=start)


class DialogueTests(unittest.TestCase):
    def test_complete_turn_windows(self):
        rows = [turn(i*2000,(i+1)*2000,speaker='CHI' if i%2 else 'MOT') for i in range(25)]
        groups = windows(rows, 50000)
        self.assertEqual([len(g) for g in groups], [10,10,5])
        self.assertEqual([r['line'] for g in groups for r in g], [r['line'] for r in rows])

    def test_overlap_missing_and_corruption_excluded(self):
        rows=[turn(0,6000),turn(5000,11000),turn(11000,17000),turn(17000,23000,corrupted=True)]
        self.assertEqual(windows(rows,23000), [])
        self.assertIsNone(spoken_text('xxx .'))
        self.assertIsNone(spoken_text('you [/] you .'))
        self.assertEqual(spoken_text("the_red_ball (.) ye:p ."), 'the red ball yep .')

    def test_verification_gate(self):
        row=dict(id='candidate',analysis_unit='all_speaker_dialogue',manual_alignment_verified=False)
        with self.assertRaises(ValueError):
            require_dialogue_verification([row])
        row['alignment_assumed_by_user']=True
        with self.assertRaises(ValueError):
            require_dialogue_verification([row])
        require_dialogue_verification([row], allow_assumed=True)
        self.assertFalse(row['manual_alignment_verified'])
        row['manual_alignment_verified']=True
        require_dialogue_verification([row])

if __name__ == '__main__':
    unittest.main()

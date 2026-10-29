import unittest
from prepare_gleason import candidates, reject_reason, clean_text

class PreparationTests(unittest.TestCase):
    def test_warning_continuations_and_overlap(self):
        text = '*CHI:\tred\n\tball . \x151000_2000\x15\n%mor:\tnoun|ball\n*MOT:\tball . \x151900_2400\x15\n@Comment:\talert! Starting at this point of the sound file\n*CHI:\tmy ball . \x152500_3500\x15'
        rows = candidates(text)
        self.assertEqual(clean_text(rows[0]['raw']), 'red ball .')
        self.assertEqual(reject_reason(rows[0], rows, 4000), 'timestamp_overlap_with_other_speaker')
        self.assertEqual(reject_reason(rows[-1], rows, 4000), 'after_explicit_corruption_warning')

    def test_bounds_and_cleaning(self):
        row = dict(span=(1000,5000), raw='my ball .', corrupted=False)
        self.assertEqual(reject_reason(row, [], 4000), 'invalid_audio_bounds')
        self.assertEqual(clean_text("it's (.) my red_ball ."), "it's my red ball .")
        self.assertIsNone(clean_text('yeah .'))

if __name__ == '__main__':
    unittest.main()

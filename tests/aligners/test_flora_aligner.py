import unittest
import os
from ontoaligner.aligner.flora.flora import FLORAAligner
from ontoaligner.aligner.flora.postprocessor import (
    flora_threshold_postprocessor,
    flora_bilateral_postprocessor,
    flora_1to1_postprocessor,
)
from ontoaligner.ontology import FLORAOMDataset
from ontoaligner.encoder import FLORAEncoder

class TestFLORAAligner(unittest.TestCase):
    def setUp(self):
        # Initializing a default FLORAAligner for testing
        self.aligner = FLORAAligner(string_identity=True)

    def test_postprocessors_basic(self):
        # Prepare dummy matchings list
        matchings = [
            {'source': '<a>', 'target': '<a>', 'score': 0.9, 'type': 'instance'},
            {'source': '<b>', 'target': '<c>', 'score': 0.2, 'type': 'instance'},
            {'source': '<c>', 'target': '<b>', 'score': 0.8, 'type': 'instance'},
        ]
        # Threshold filter at 0.5 should remove the (b,c) pair
        thr = flora_threshold_postprocessor(matchings, prefix1='', prefix2='', threshold=0.5)
        # instance_alignments, class_alignments, predicate_alignments

        self.assertEqual(len(thr), 3)
        self.assertEqual(len(thr[0]), 2) # instance_alignments
        self.assertEqual(len(thr[1]), 2) # class_alignments
        self.assertEqual(len(thr[2]), 0) # predicate_alignments


        # Bilateral: only keep pairs where source-target are mutual best
        bilat = flora_bilateral_postprocessor(
            same_as_scores={m['source']: {m['target']: m['score']} for m in matchings},
            source_prefix=''
        )
        # For our dummy data, 'a' matches to 'a' only
        self.assertIn({'source': 'a', 'target': 'a', 'score': 0.9}, bilat)

        # One-to-one: enforce unique assignments
        one2one = flora_1to1_postprocessor(
            same_as_scores={m['source']: {m['target']: m['score']} for m in matchings}
        )
        # Should contain at most one mapping per source
        sources = [m['source'] for m in one2one]
        self.assertEqual(len(sources), len(set(sources)))

    def test_mini_set_end_to_end(self):
        """
        End-to-end test on the flora-mini-set: should align Elvis and marriedTo relation.
        """
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'flora-mini-set'))
        src = os.path.join(base, 'mini1.ttl')
        tgt = os.path.join(base, 'mini2.ttl')
        # Parse dataset
        task = FLORAOMDataset(ontology_name='flora-mini-set')
        dataset = task.collect(
            source_ontology_path=src,
            target_ontology_path=tgt
        )
        # Encode graphs
        graphs = FLORAEncoder()(source=dataset['source'], target=dataset['target'])
        # Align
        aligner = FLORAAligner(string_identity=True)
        matchings = aligner.generate(input_data=graphs)
        # Assert Elvis entity alignment
        self.assertTrue(any(m['source'].endswith('Elvis') and m['target'].endswith('Elvis') for m in matchings), f"Expected Elvis->Elvis in {matchings}")
        # Assert predicate mapping for 'marriedTo'
        preds = aligner.get_predicate2super_predicate()
        # Find the marriedTo key
        key = next((k for k in preds if 'marriedTo' in k), None)
        self.assertIsNotNone(key, f"No predicate key containing 'marriedTo' found in {list(preds.keys())}")
        target_map = preds[key]
        self.assertTrue(any('marriedTo' in t and score > 0 for t, score in target_map.items()), f"Expected non-zero mapping for 'marriedTo' in {target_map}")

if __name__ == '__main__':
    unittest.main()

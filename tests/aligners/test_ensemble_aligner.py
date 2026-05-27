import pytest

from ontoaligner.base import BaseEncoder, BaseOMModel
from ontoaligner.aligner.ensemble import EnsembleAligner
from ontoaligner.aligner.ensemble.voting import (
    ReciprocalRankFusionVoting,
    BordaCountVoting,
    CondorcetVoting,
    ScoreAverageVoting,
    WeightedVoting
)
from ontoaligner import AlignerPipeline


class DummyEncoder(BaseEncoder):
    """Dummy encoder for ensemble tests."""

    def parse(self, **kwargs):
        return [kwargs["source"], kwargs["target"]]

    def get_encoder_info(self):
        return "DUMMY ENCODER"


class DummyAligner(BaseOMModel):
    """Dummy aligner for ensemble tests."""

    def __str__(self):
        return "DummyAligner"

    def generate(self, input_data):
        source_onto, target_onto = input_data
        return [
            {
                "source": source_onto[0]["iri"],
                "target": target_onto[0]["iri"],
                "score": 0.9,
            }
        ]


class DummyGroupedAligner(BaseOMModel):
    """Dummy retrieval-style aligner for ensemble tests."""

    def __str__(self):
        return "DummyGroupedAligner"

    def generate(self, input_data):
        source_onto, target_onto = input_data
        return [
            {
                "source": source_onto[0]["iri"],
                "target-cands": [target_onto[0]["iri"], target_onto[1]["iri"]],
                "score-cands": [0.9, 0.7],
            }
        ]


class StaticBranch:
    """Branch that returns predefined predictions."""

    def __init__(self, predictions):
        self.predictions = predictions

    def generate(self, input_data=None):
        return self.predictions


@pytest.fixture
def om_dataset():
    return {
        "source": [{"iri": "s1", "text": "alpha"}],
        "target": [
            {"iri": "t1", "text": "alpha"},
            {"iri": "t2", "text": "beta"},
        ],
        "reference": [],
    }


def score_filter(predicts, threshold=0.5):
    return [prediction for prediction in predicts if prediction["score"] >= threshold]


def test_aligner_branch_generates_predictions(om_dataset):
    branch = AlignerPipeline(
        encoder=DummyEncoder(),
        aligner=DummyAligner(),
        om_dataset=om_dataset,
    )

    predictions = branch.generate()

    assert len(predictions) == 1
    assert predictions[0]["source"] == "s1"
    assert predictions[0]["target"] == "t1"
    assert predictions[0]["score"] == 0.9


def test_aligner_branch_with_postprocessor(om_dataset):
    branch = AlignerPipeline(
        encoder=DummyEncoder(),
        aligner=DummyAligner(),
        om_dataset=om_dataset,
        postprocessor=score_filter,
        postprocessor_params={"threshold": 0.95},
    )

    predictions = branch.generate()

    assert predictions == []


def test_ensemble_aligner_requires_multiple_branches():
    branch = StaticBranch([{"source": "s1", "target": "t1", "score": 1.0}])

    with pytest.raises(ValueError):
        EnsembleAligner(branches=[("single", branch, 1.0)])


def test_ensemble_aligner_combines_flat_outputs():
    branch_1 = StaticBranch(
        [
            {"source": "s1", "target": "t1", "score": 0.9},
            {"source": "s1", "target": "t2", "score": 0.8},
        ]
    )
    branch_2 = StaticBranch(
        [
            {"source": "s1", "target": "t1", "score": 0.7},
        ]
    )

    ensemble = EnsembleAligner(
        branches=[
            ("branch-1", branch_1, 1.0),
            ("branch-2", branch_2, 1.0),
        ],
        voting=ReciprocalRankFusionVoting(k=60),
    )

    predictions = ensemble.generate()

    assert len(predictions) == 2
    assert predictions[0]["target"] == "t1"


def test_ensemble_aligner_flattens_grouped_outputs(om_dataset):
    branch_1 = AlignerPipeline(
        encoder=DummyEncoder(),
        aligner=DummyGroupedAligner(),
        om_dataset=om_dataset,
    )
    branch_2 = StaticBranch(
        [
            {"source": "s1", "target": "t1", "score": 1.0},
        ]
    )

    ensemble = EnsembleAligner(
        branches=[
            ("grouped", branch_1, 1.0),
            ("flat", branch_2, 1.0),
        ],
        voting=ReciprocalRankFusionVoting(k=60),
    )

    predictions = ensemble.generate()

    assert len(predictions) == 2
    assert predictions[0]["target"] == "t1"


@pytest.mark.parametrize(
    "voting",
    [
        ReciprocalRankFusionVoting(k=60),
        BordaCountVoting(),
        CondorcetVoting(),
        ScoreAverageVoting(),
        WeightedVoting(min_votes=1),
    ],
)
def test_voting_methods_generate_predictions(voting):
    branch_outputs = [
        (
            [
                {"source": "s1", "target": "t1", "score": 0.9},
                {"source": "s1", "target": "t2", "score": 0.8},
            ],
            1.0,
        ),
        (
            [
                {"source": "s1", "target": "t1", "score": 0.7},
                {"source": "s1", "target": "t3", "score": 0.6},
            ],
            1.0,
        ),
    ]

    predictions = voting.combine(branch_outputs=branch_outputs)

    assert len(predictions) > 0
    assert all("source" in prediction for prediction in predictions)
    assert all("target" in prediction for prediction in predictions)
    assert all("score" in prediction for prediction in predictions)


def test_weighted_voting_min_votes_filters_predictions():
    branch_outputs = [
        (
            [
                {"source": "s1", "target": "t1", "score": 0.9},
                {"source": "s1", "target": "t2", "score": 0.8},
            ],
            1.0,
        ),
        (
            [
                {"source": "s1", "target": "t1", "score": 0.7},
            ],
            1.0,
        ),
    ]

    voting = WeightedVoting(min_votes=2)
    predictions = voting.combine(branch_outputs=branch_outputs)

    assert len(predictions) == 1
    assert predictions[0]["target"] == "t1"


def test_voting_requires_score():
    branch_outputs = [
        (
            [
                {"source": "s1", "target": "t1"},
            ],
            1.0,
        ),
        (
            [
                {"source": "s1", "target": "t2", "score": 0.5},
            ],
            1.0,
        ),
    ]

    voting = ReciprocalRankFusionVoting(k=60)

    with pytest.raises(KeyError):
        voting.combine(branch_outputs=branch_outputs)

import pytest
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.ontology import GenericOMDataset


def test_pipeline_initialization():
    """Test that Pipeline can be initialized."""
    pipeline = OntoAlignerPipeline(
        task_class=GenericOMDataset,
        source_ontology_path="tests/data/test-case1.owl",
        target_ontology_path="tests/data/test-case1.owl",
        reference_matching_path=None,
    )
    assert pipeline is not None


def test_pipeline_with_lightweight_aligner(sample_ontology, temp_output_dir):
    """Test pipeline with lightweight aligner."""
    pipeline = OntoAlignerPipeline(
        task_class=GenericOMDataset,
        source_ontology_path="tests/data/test-case1.owl",
        target_ontology_path="tests/data/test-case1.owl",
        reference_matching_path=None,
        output_dir=str(temp_output_dir),
    )

    result = pipeline(
        method="lightweight",
        fuzzy_sm_threshold=0.5,
        save_matchings=True,
        output_file_name="alignment_result",
    )

    assert result is not None


@pytest.mark.skip(reason="Requires OpenAI API key")
def test_pipeline_with_rag_aligner(sample_ontology, temp_output_dir):
    """Test pipeline with RAG aligner."""
    pipeline = OntoAlignerPipeline(
        task_class=GenericOMDataset,
        source_ontology_path="tests/data/test-case1.owl",
        target_ontology_path="tests/data/test-case1.owl",
        reference_matching_path=None,
        output_dir=str(temp_output_dir),
    )

    result = pipeline(
        method="rag", save_matchings=True, output_file_name="rag_alignment_result"
    )

    assert result is not None
    output_path = temp_output_dir / "rag_alignment_result.xml"
    assert output_path.exists()


def test_pipeline_with_invalid_aligner():
    """Test pipeline with invalid aligner type."""
    pipeline = OntoAlignerPipeline(
        task_class=GenericOMDataset,
        source_ontology_path="tests/data/test-case1.owl",
        target_ontology_path="tests/data/test-case1.owl",
        reference_matching_path=None,
    )

    with pytest.raises(ValueError):
        pipeline(method="invalid_aligner")

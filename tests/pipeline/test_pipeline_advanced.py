import pytest
import os
import shutil
from pathlib import Path
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.ontology import GenericOMDataset


@pytest.fixture
def complex_pipeline():
    """Create a pipeline with complex configuration."""
    return OntoAlignerPipeline(
        task_class=GenericOMDataset,
        source_ontology_path="tests/data/test-case1.owl",
        target_ontology_path="tests/data/test-case1.owl",
        reference_matching_path=None,
        output_dir="test_output",
    )


def test_pipeline_error_recovery(complex_pipeline):
    """Test pipeline's error recovery capabilities."""
    # Test with invalid method
    with pytest.raises(ValueError):
        complex_pipeline(method="invalid_method")

    # Test with invalid threshold
    with pytest.raises(ValueError):
        complex_pipeline(method="lightweight", fuzzy_sm_threshold=2.0)

    # Test with missing output directory
    shutil.rmtree("test_output", ignore_errors=True)
    result = complex_pipeline(
        method="lightweight", save_matchings=True, output_file_name="test.xml"
    )
    assert result is not None
    assert Path("test_output/test.xml").exists()


def test_pipeline_resource_cleanup(complex_pipeline):
    """Test proper resource cleanup after pipeline execution."""
    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Run pipeline multiple times
    for i in range(5):
        result = complex_pipeline(
            method="lightweight",
            fuzzy_sm_threshold=0.5,
            save_matchings=True,
            output_file_name=f"cleanup_test_{i}.xml",
        )
        assert result is not None

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Check that memory usage increase is reasonable
    assert memory_increase < 200 * 1024 * 1024  # 200MB in bytes


def test_pipeline_large_ontology(tmp_path):
    """Test pipeline with large ontologies."""
    # Create large test ontologies
    source_onto = create_large_ontology(tmp_path / "source_large.owl", 1000)
    target_onto = create_large_ontology(tmp_path / "target_large.owl", 1000)

    pipeline = OntoAlignerPipeline(
        task_class=GenericOMDataset,
        source_ontology_path=str(source_onto),
        target_ontology_path=str(target_onto),
        reference_matching_path=None,
        output_dir=str(tmp_path / "output"),
    )

    result = pipeline(
        method="lightweight",
        fuzzy_sm_threshold=0.5,
        save_matchings=True,
        output_file_name="large_test.xml",
    )

    assert result is not None
    assert len(result) > 0  # Should have some matches


def create_large_ontology(file_path, num_classes):
    """Helper function to create large test ontologies."""
    from rdflib import Graph, Literal, Namespace
    from rdflib.namespace import RDF, RDFS, OWL

    g = Graph()
    ns = Namespace("http://example.org/")

    for i in range(num_classes):
        class_uri = ns[f"Class_{i}"]
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(f"Test Class {i}")))

        if i > 0:
            g.add((class_uri, RDFS.subClassOf, ns[f"Class_{i-1}"]))

    g.serialize(destination=str(file_path), format="xml")
    return file_path

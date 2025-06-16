import pytest
import shutil
from pathlib import Path
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.ontology import GenericOMDataset
import sys
import gc


@pytest.fixture
def complex_pipeline():
    """Create a pipeline with complex configuration."""
    return OntoAlignerPipeline(
        task_class=GenericOMDataset,
        source_ontology_path="tests/data/test-case1.owl",
        target_ontology_path="tests/data/test-case1.owl",
        reference_matching_path=None,
        output_dir="test_output",
        output_format="xml"
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
        method="lightweight", save_matchings=True, output_file_name="test"
    )
    assert result is not None
    assert Path("test_output/lightweight/test.xml").exists()

def get_total_size(objects):
    """Estimate total size of objects in memory."""
    seen = set()
    size = 0
    for obj in objects:
        if id(obj) not in seen:
            seen.add(id(obj))
            try:
                size += sys.getsizeof(obj)
            except TypeError:
                pass  # Some built-in objects don't support getsizeof
    return size

def test_pipeline_resource_cleanup(complex_pipeline):
    """Test for potential memory leaks without external libraries."""
    gc.collect()
    initial_objects = gc.get_objects()
    initial_size = get_total_size(initial_objects)

    for i in range(5):
        result = complex_pipeline(
            method="lightweight",
            fuzzy_sm_threshold=0.5,
            save_matchings=True,
            output_file_name=f"cleanup_test_{i}",
        )
        assert result is not None

    gc.collect()
    final_objects = gc.get_objects()
    final_size = get_total_size(final_objects)

    # Print difference (you can assert if needed)
    print(f"Initial size: {initial_size / 1024:.2f} KB")
    print(f"Final size: {final_size / 1024:.2f} KB")
    print(f"Size difference: {(final_size - initial_size) / 1024:.2f} KB")

    # Allow some leeway (e.g. 10 MB) for object growth
    assert (final_size - initial_size) < 10 * 1024 * 1024  # 10MB in bytes


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
        output_file_name="large_test",
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

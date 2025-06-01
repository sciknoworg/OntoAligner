import pytest
from ontoaligner.ontology import GenericOntology
from rdflib import Graph


def test_ontology_loading(sample_ontology):
    """Test that ontology can be loaded successfully."""
    assert sample_ontology is not None
    assert isinstance(sample_ontology, Graph)

def test_invalid_ontology_path():
    """Test that loading invalid ontology path raises error."""
    with pytest.raises(Exception):
        GenericOntology().load_ontology("nonexistent.owl")

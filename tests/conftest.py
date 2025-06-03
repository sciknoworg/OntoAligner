import pytest
from pathlib import Path
from ontoaligner.ontology import GenericOntology


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_ontology(test_data_dir):
    ontology_path = test_data_dir / "test-case1.owl"
    return GenericOntology().load_ontology(str(ontology_path))


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

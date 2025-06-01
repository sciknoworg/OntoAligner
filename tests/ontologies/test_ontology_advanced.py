import pytest
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL
from ontoaligner.ontology import GenericOntology


@pytest.fixture
def large_ontology():
    """Create a large test ontology with many classes and relationships."""
    g = Graph()
    ns = Namespace("http://example.org/")

    # Create 100 classes with relationships
    for i in range(100):
        class_uri = ns[f"Class_{i}"]
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(f"Test Class {i}")))

        # Add subclass relationships
        if i > 0:
            g.add((class_uri, RDFS.subClassOf, ns[f"Class_{i-1}"]))

        # Add some comments and synonyms
        g.add((class_uri, RDFS.comment, Literal(f"This is test class number {i}")))
        g.add((class_uri, RDFS.label, Literal(f"Alternative Name {i}")))

    return g

def test_invalid_ontology_format():
    """Test handling of invalid ontology formats."""
    with pytest.raises(Exception):
        ontology = GenericOntology()
        ontology.parse("nonexistent.owl")


def test_ontology_modification():
    """Test modifying ontology data after loading."""
    ontology = GenericOntology()

    # Create a simple test graph
    g = Graph()
    ns = Namespace("http://example.org/")
    class_uri = ns["TestClass"]
    g.add((class_uri, RDF.type, OWL.Class))
    g.add((class_uri, RDFS.label, Literal("Test Class")))

    # Add the graph to the ontology
    ontology.graph = g
    data = ontology.extract_data(g)

    # Verify initial state
    assert len(data) == 1
    assert data[0]["label"] == "Test Class"

    # Modify the graph
    g.add((class_uri, RDFS.comment, Literal("New comment")))

    # Re-extract data and verify changes
    updated_data = ontology.extract_data(g)
    assert len(updated_data) == 1
    assert "New comment" in updated_data[0]["comment"]


def test_special_characters():
    """Test handling of special characters in ontology data."""
    ontology = GenericOntology()

    # Create a test graph with special characters
    g = Graph()
    ns = Namespace("http://example.org/")
    special_chars = ["é", "ñ", "ß", "漢", "🌟"]

    for i, char in enumerate(special_chars):
        class_uri = ns[f"Class_{i}"]
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(f"Test Class {char}")))

    # Parse and verify
    ontology.graph = g
    data = ontology.extract_data(g)

    assert len(data) == len(special_chars)
    for i, char in enumerate(special_chars):
        assert f"Test Class {char}" in [item["label"] for item in data]


def test_circular_references():
    """Test handling of circular references in ontology."""
    ontology = GenericOntology()

    # Create a graph with circular references
    g = Graph()
    ns = Namespace("http://example.org/")

    # Create circular subclass relationship
    class_a = ns["ClassA"]
    class_b = ns["ClassB"]
    class_c = ns["ClassC"]

    for class_uri in [class_a, class_b, class_c]:
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(str(class_uri))))

    # Create circular reference: A -> B -> C -> A
    g.add((class_a, RDFS.subClassOf, class_b))
    g.add((class_b, RDFS.subClassOf, class_c))
    g.add((class_c, RDFS.subClassOf, class_a))

    # Should handle circular references without infinite recursion
    ontology.graph = g
    data = ontology.extract_data(g)

    assert len(data) == 3
    # Verify each class has both parent and child relationships
    for item in data:
        assert len(item["parents"]) > 0
        assert len(item["childrens"]) > 0

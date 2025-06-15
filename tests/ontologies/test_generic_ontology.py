import unittest
import os
from rdflib import URIRef, RDFS, OWL, RDF, Graph
from ontoaligner.ontology.generic import GenericOntology
import ontoaligner


class TestGenericOntology(unittest.TestCase):
    def setUp(self):
        self.ontology = GenericOntology()
        self.test_graph = Graph()

        # Load test-case1.owl
        test_file_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test-case1.owl"
        )
        self.test_graph.parse(test_file_path)

        # Define URIs from test-case1.owl
        self.animal_uri = URIRef("http://example.org/Animal")
        self.mammal_uri = URIRef("http://example.org/Mammal")
        self.dog_uri = URIRef("http://example.org/Dog")

        # Add explicit owl:Class type to Animal since it's the root
        self.test_graph.add((self.animal_uri, RDF.type, OWL.Class))

        self.ontology.graph = self.test_graph

    def test_get_label(self):
        """Test label extraction with different scenarios"""
        # Test with existing label
        self.assertEqual(self.ontology.get_label(str(self.mammal_uri)), "Mammal")
        self.assertEqual(self.ontology.get_label(str(self.dog_uri)), "Dog")

        # Test with non-existent class
        self.assertEqual(
            self.ontology.get_label("http://example.org/NonExistent"),
            "NonExistent",  # The method returns the last part of the URI if no label found
        )

        # Test with URI fragment
        self.assertEqual(
            self.ontology.get_label("http://example.org/TestClass#Fragment"), "Fragment"
        )

        # Test with URI path
        self.assertEqual(
            self.ontology.get_label("http://example.org/path/LastPart"), "LastPart"
        )

    def test_get_synonyms(self):
        """Test synonym extraction"""
        # No synonyms in test-case1.owl, should return empty list
        synonyms = self.ontology.get_synonyms(self.mammal_uri)
        self.assertEqual(len(synonyms), 0)

    def test_get_parents(self):
        """Test parent class extraction"""
        # Test Dog's parent (Mammal) - Dog has subClassOf relationship
        parents = self.ontology.get_parents(self.dog_uri)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["iri"], str(self.mammal_uri))
        self.assertEqual(parents[0]["label"], "Mammal")

    def test_get_childrens(self):
        """Test children class extraction"""
        # Test Animal's children (Mammal)
        children = self.ontology.get_childrens(self.animal_uri)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["label"], "Mammal")

        # Test Mammal's children (Dog)
        children = self.ontology.get_childrens(self.mammal_uri)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["label"], "Dog")

    def test_get_comments(self):
        """Test comment extraction"""
        # No comments in test-case1.owl, should return empty list
        comments = self.ontology.get_comments(self.mammal_uri)
        self.assertEqual(len(comments), 0)

    def test_get_class_info(self):
        """Test complete class info extraction"""
        # Test Dog class info - Dog has subClassOf relationship
        class_info = self.ontology.get_class_info(self.dog_uri)
        self.assertIsNotNone(class_info)
        self.assertEqual(class_info["label"], "Dog")
        self.assertEqual(len(class_info["childrens"]), 0)  # No children
        self.assertEqual(len(class_info["parents"]), 1)  # Mammal
        self.assertEqual(len(class_info["synonyms"]), 0)  # No synonyms
        self.assertEqual(len(class_info["comment"]), 0)  # No comments

    def test_generic_ontology_parser(self):
        """Test that the parse function loads an ontology correctly."""
        ontology = ontoaligner.ontology.GenericOntology()
        ontology_path = os.path.join(
            os.path.dirname(__file__), "..", "data/test-case1.owl"
        )
        data = ontology.parse(ontology_path)

        # Ensure parsed ontology data is not empty
        self.assertGreater(len(data), 0)

        # Check expected subclass relationships
        mammal = URIRef("http://example.org/Mammal")
        animal = URIRef("http://example.org/Animal")
        self.assertTrue((mammal, RDFS.subClassOf, animal) in ontology.graph)

        # Check expected labels
        label_predicate = RDFS.label
        expected_label = "Mammal"
        found_labels = [
            str(o) for s, p, o in ontology.graph if s == mammal and p == label_predicate
        ]
        self.assertIn(expected_label, found_labels)


if __name__ == "__main__":
    unittest.main()

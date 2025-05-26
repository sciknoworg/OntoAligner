import unittest
import os
import ontoaligner
from rdflib import URIRef, RDFS

class TestOntology(unittest.TestCase):

    def test_generic_ontology_parser(self):
        """Test that the parse function loads an ontology correctly."""
        ontology = ontoaligner.ontology.GenericOntology()
        ontology_path = os.path.join(os.path.dirname(__file__), "data/test-case1.owl")
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
        found_labels = [str(o) for s, p, o in ontology.graph if s == mammal and p == label_predicate]
        self.assertIn(expected_label, found_labels)


if __name__ == '__main__':
    unittest.main()

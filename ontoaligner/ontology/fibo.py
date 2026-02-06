from rdflib import Graph, URIRef, RDFS, SKOS, BNode, Namespace
from rdflib.namespace import OWL, RDF
from tqdm import tqdm
from typing import Any, Dict, List

from ...base import OMDataset
from ..generic import GenericOntology


track = "FIBO"

class FIBOOntology(GenericOntology):
    """
    An class for parsing the Financial Industry Business Ontology (FIBO) and compatible ontologies. 
    This class defines methods to extract data such as  names, labels, IRIs, children, parents, synonyms, 
    and comments for ontology classes.
    """

    def __init__(self, language: str = 'en'):
        self.language = language
        self.cmns_av = Namespace("https://www.omg.org/spec/Commons/AnnotationVocabulary/")


    def get_label(self, owl_class: str) -> Any:
        """
        Extracts the label for a given URI in the specified language from the RDF graph.
        If no valid label is found, returns None.

        :param uri: URI of the entity to retrieve the label for.
        :return: The label in the specified language, or None if no label found.
        """
        entity = URIRef(owl_class)
        labels = list(self.graph.objects(subject=entity, predicate=RDFS.label))
        for label in labels:
            if hasattr(label, 'language') and label.language == self.language:
                return self.is_valid_label(str(label))
        if labels:
            first_label = str(labels[0])
            if not first_label.startswith("http"):
                return self.is_valid_label(first_label)
        if "#" in owl_class:
            local_name = owl_class.split("#")[-1]
        elif "/" in owl_class:
            local_name = owl_class.split("/")[-1]
        else:
            local_name = owl_class
        label = self.is_valid_label(local_name)
        if not label:
            return None
        return label

    def get_synonyms(self, owl_class: URIRef) -> List[Dict[str, str]]:
        """
        Retrieves synonyms for the given ontology class.

        Tries different properties (like `rdfs:label`, `skos:altLabel`).
        """
        synonyms = []

        # for pref_label in self.graph.objects(owl_class, SKOS.prefLabel):     
        #     print(pref_label)

        for synonym in self.graph.objects(owl_class, self.cmns_av.synonym):
            synonyms.append(str(synonym))
            # synonyms.append({
            #     "iri": str(owl_class),
            #     "label": str(synonym),
            #     "name": str(synonym),
            # })
        # Search for alternative labels (SKOS altLabel)
        for alt_label in self.graph.objects(owl_class, SKOS.altLabel):
            synonyms.append(str(alt_label))
            # synonyms.append({
            #     "iri": str(owl_class),
            #     "label": str(alt_label),
            #     "name": str(alt_label),
            # })
        return synonyms

    def get_comments(self, owl_class: URIRef) -> List[str]:
        """
        Retrieves comments for the given class. 
        Treat comments as definitions and include skos:definition and cmns-av:explanatoryNote
        The objective is to provide a longer textual descriptions of the class for better matching.
        """
        comments = []

        for comment in self.graph.objects(owl_class, RDFS.comment):
            comments.append(str(comment))

        # add skos:definitons as comments    
        for definition in self.graph.objects(owl_class, SKOS.definition):
            comments.append(str(definition))

        # add common annotation vocabulary as comments
        for explanatory_note in self.graph.objects(owl_class, self.cmns_av.explanatoryNote):
            comments.append(str(explanatory_note))

        return comments


    def get_class_info(self, owl_class: URIRef) -> Any:
        """
        Collects all relevant information for a given ontology class.
        """
        label = self.get_label(owl_class)
        name = self.get_name(owl_class)
        iri = self.get_iri(owl_class)

        if not label or not iri:
            return None

        class_info = {
            "name": name,
            "iri": iri,
            "label": label,            
            "childrens": self.get_childrens(owl_class),
            "parents": self.get_parents(owl_class),
            "synonyms": self.get_synonyms(owl_class),
            "comment": self.get_comments(owl_class)
        }
        return class_info
    
class FIBOOMDataset(OMDataset):
    """
    A dataset class for matching data against FIBO.
    """
    track = track
    ontology_name = "Source-FIBO"
    source_ontology = FIBOOntology()
    target_ontology = FIBOOntology()
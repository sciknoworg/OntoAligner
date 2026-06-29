Parsers
==================

.. hint::

    Different aligner models may require specific types of parsers and encoders. The **Aligners** section clearly outlines these dependencies, detailing which parser and encoder components are used by each aligner type.

The ``parser`` module in OntoAligner provides essential ``oaei`` (track ontologies) and ``generic`` ontology parsers for handling ontologies for ontology alignment tasks. This tutorial explains the structure, key components, and how to utilize these modules in your ontology alignment workflows.

Usage
-----------------------

To be able to run an alignment task, you need to create an ``OMDataset`` for your work. The ``OMDataset`` class is responsible for managing ontology matching datasets, handling the source and target ontologies, and parsing reference alignments. This class utilizes ontology parsers for parsing ontologies and ``BaseAlignmentsParser`` for handling reference alignments, allowing users to define custom datasets by specifying track names, ontology names, and parsing methods.

.. code-block:: python

    class OMDataset(ABC):
        track: str = ""
        ontology_name: str = ""

        source_ontology: Any = None
        target_ontology: Any = None

        alignments: Any = BaseAlignmentsParser()

        def collect(self, source_ontology_path: str, target_ontology_path: str, reference_matching_path: str="") -> Dict:
            ....

Now, for specifying source and target ontologies, we provide two components: ``generic`` and ``oaei``. Both modules enables loading various ontologies and formatting them into the following structure:

.. code-block::

    [{
        'name': 'PhaseEquilibrium',
        'iri': 'http://matonto.org/ontologies/matonto#PhaseEquilibrium',
        'label': 'Phase Equilibrium',
        'childrens': [],
        'parents': [{'iri': 'http://ontology.dumontierlab.com/MeasuredProperty',
                     'name': 'MeasuredProperty',
                     'label': 'measured property'}],
        'synonyms': [],
        'comment': ['The conditions at which two phases can be at equilibrium']
   }, ... ]

In the final ``OMDataset`` will form a parsed ontology alignment task using source and target ontologies in the following format:

.. code-block::

    {
        "dataset-info": {
            "track": "track name",
            "ontology-name": "source ontology name-target ontology name"
        },
        "source": [
            {
                "name": "iri name",
                "iri": "iri",
                "label": "label",
                "childrens": [{"iri": "", "name":"", "label":""}, ... ],
                "parents": [{"iri": "", "name":"", "label":""}, ... ],
                "synonyms": ["synonym1", ...],
                "comment": ["comment1",... ]
            }
            ...
        ],
        "target": [
            {
                "name": "iri name",
                "iri": "iri",
                "label": "label",
                "childrens": [{"iri": "", "name":"", "label":""}, ... ],
                "parents": [{"iri": "", "name":"", "label":""}, ... ],
                "synonyms": ["synonym1", ...],
                "comment": ["comment1",... ]
            }
            ...
        ],
        "reference": [
            {
                "source": "source iri",
                "target": "target iri",
                "relation": "="
            },
            ...
        ]
    }

.. hint::
    If you don't specify the ``reference_matching_path`` in the ``OMDataset``, it will be assumed to be an empty list ``[]``.


Generic Parser
-------------------------

.. sidebar:: Materials:

    * `Download the Conference ontology <http://www.scholarlydata.org/ontology/doc/>`__
    * `Download the GEO ontology <http://purl.obolibrary.org/obo/geo.owl>`__
    * `Download the GeoNames ontology: <https://www.geonames.org/ontology/documentation.html>`__



An generic class for parsing OWL/rdf based ontologies. This class defines methods to extract data such as names, labels, IRIs, children, parents, synonyms, and comments for ontology classes. It provides a smooth parser for given ontology on the hand which later can be used for ontology alignment. To use this module for desired ontology you need to use the following code:


.. code-block:: python

    from ontoaligner.ontology import GenericOntology
    ontology =  GenericOntology()
    parsed_ontology = ontology.parse("conference.owl")



As another example, suppose you want to perform ontology alignment for the ``GEO`` and ``GeoNames`` ontologies. In this case, you can use the ``GenericOMDataset`` as follows:

.. code-block:: python

    from ontoaligner.ontology import GenericOMDataset
    task =  GenericOMDataset(
        track = "Geographical"         # optional
        ontology_name = "GEO-GeoNames" # optional
    )
    dataset = task.collect(source_ontology_path="geo.owl", target_ontology_path="geonames.owl")

Parser Customization
-------------------------

.. sidebar:: Related Links:

    * `GenericOntology source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/ontology/generic.py#L23-L195>`__
    * `BaseOntologyParser source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/base/ontology.py#L33-L259>`__
    * `OMDataset source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/base/dataset.py#L30-L112>`__

The ``generic`` parser covers the common case where an OWL/RDF ontology can be loaded
with ``rdflib`` and concept information can be extracted from standard properties such as
``rdfs:label``, ``skos:altLabel``, ``rdfs:comment``, and ``rdfs:subClassOf``.

In practice, ontologies may store the same information differently. For example, an
ontology may use ``skos:prefLabel`` as the main label, store definitions under
``dcterms:description``, use domain-specific synonym properties, or require additional
metadata such as normalized labels, URI fragments, or enriched text for downstream
encoders and aligners.

For these cases, OntoAligner parsers can be customized by extending the existing parser
classes and overriding only the part that differs.

.. hint::

    If your ontology is still an OWL/RDF ontology, start by extending
    :class:`~ontoaligner.ontology.generic.GenericOntology`. If the source is not an
    OWL/RDF ontology, or if the loading and iteration logic is completely different,
    extend :class:`~ontoaligner.base.ontology.BaseOntologyParser` directly.


.. tab:: 🔁 Parser Flow

    The :class:`~ontoaligner.ontology.generic.GenericOntology` parser follows a small
    parsing flow. The ontology is loaded first, then each ontology class is visited, and
    finally one dictionary is created for each parsed concept.

    .. code-block:: text

        parse(input_file_path)
            ├── load_ontology(input_file_path)
            ├── extract_data(graph)
            └── get_class_info(owl_class)

    The ``get_class_info`` method is the assembly point. It calls smaller hook methods
    such as ``get_label``, ``get_synonyms``, ``get_comments``, ``get_parents``, and
    ``get_childrens`` to create the final concept dictionary.

    .. list-table::
        :header-rows: 1
        :widths: 30 70

        * - Method
          - What it controls
        * - ``load_ontology``
          - How the ontology file is loaded. Override this if the file needs custom loading.
        * - ``extract_data``
          - Which ontology entities are parsed. Override this only if you need to iterate over something other than the default class set.
        * - ``get_label``
          - Which value becomes the main concept ``label``.
        * - ``is_valid_label``
          - Which labels should be ignored, such as generic placeholders.
        * - ``get_synonyms``
          - Which values are collected as alternative names.
        * - ``get_comments``
          - Which values are collected as comments, descriptions, or definitions.
        * - ``get_parents``
          - How parent concepts are collected.
        * - ``get_childrens``
          - How child concepts are collected.
        * - ``is_class``
          - Which RDF resources should be treated as valid ontology classes.
        * - ``get_class_info``
          - How the final parsed concept dictionary is assembled.


.. tab:: 🎯 Customization Guide

    Before writing a custom parser, inspect the ontology and identify where the information
    is stored. Check whether labels are stored in ``rdfs:label`` or ``skos:prefLabel``,
    whether synonyms are stored in ``skos:altLabel`` or another property, and whether
    definitions are stored in ``rdfs:comment`` or a custom annotation property.

    Use the smallest customization that solves the problem.

    .. list-table::
        :header-rows: 1
        :widths: 40 60

        * - Requirement
          - Recommended customization
        * - The ontology uses different label properties.
          - Override ``get_label``.
        * - The ontology uses different synonym properties.
          - Override ``get_synonyms``.
        * - The ontology stores definitions in another property.
          - Override ``get_comments``.
        * - The hierarchy does not use the expected subclass relation.
          - Override ``get_parents`` and ``get_childrens``.
        * - You need extra metadata for later components.
          - Override ``get_class_info`` and add a nested custom field.
        * - The ontology uses another RDF serialization.
          - Override ``load_ontology``.
        * - The source is not an RDF/OWL ontology.
          - Extend ``BaseOntologyParser`` directly.

    .. note::

        Ontology parsers handle source and target ontologies. Reference matching files are
        parsed separately by :class:`~ontoaligner.base.ontology.BaseAlignmentsParser`.
        For example, the BioML parser customizes TSV reference alignment parsing in
        `bioml.py <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/ontology/oaei/bioml.py#L60-L93>`__.


.. tab::  🔀 Redirect Fields

    The simplest customization keeps the same output structure and only changes where
    values come from.

    The following parser reads labels from ``skos:prefLabel``, synonyms
    from ``skos:altLabel`` and ``skos:hiddenLabel``, and comments from both
    ``rdfs:comment`` and ``dcterms:description``.

    .. code-block:: python

        from typing import Dict, List

        from rdflib import Literal, URIRef
        from rdflib.namespace import DCTERMS, RDFS, SKOS

        from ontoaligner.base import OMDataset
        from ontoaligner.ontology import GenericOntology


        class MyOntology(GenericOntology):
            """
            A custom parser that redirects labels, synonyms, and comments to the
            properties used by a specific ontology.
            """

            def _get_literal_values(
                    self,
                    owl_class: URIRef,
                    predicates: List[URIRef],
            ) -> List[str]:
                values = []
                for predicate in predicates:
                    for node in self.graph.objects(owl_class, predicate):
                        if isinstance(node, Literal):
                            value = str(node).strip()
                            if value:
                                values.append(value)
                return values

            def get_label(self, owl_class: URIRef) -> str:
                """
                Uses skos:prefLabel as the preferred label and falls back to the
                default GenericOntology label logic.
                """
                fallback_labels = []

                for node in self.graph.objects(owl_class, SKOS.prefLabel):
                    if not isinstance(node, Literal):
                        continue

                    label = self.is_valid_label(str(node).strip())
                    if not label:
                        continue

                    if getattr(node, "language", None) == self.language:
                        return label

                    if getattr(node, "language", None) is None:
                        fallback_labels.append(label)

                if fallback_labels:
                    return fallback_labels[0]

                return super().get_label(owl_class)

            def get_synonyms(self, owl_class: URIRef) -> List[Dict[str, str]]:
                """
                Reads synonyms from skos:altLabel and skos:hiddenLabel.
                """
                synonyms = []
                values = self._get_literal_values(
                    owl_class=owl_class,
                    predicates=[SKOS.altLabel, SKOS.hiddenLabel],
                )

                for value in values:
                    synonyms.append({
                        "iri": str(owl_class),
                        "label": value,
                        "name": value,
                    })

                return synonyms

            def get_comments(self, owl_class: URIRef) -> List[str]:
                """
                Reads comments and definitions from multiple description properties.
                """
                return self._get_literal_values(
                    owl_class=owl_class,
                    predicates=[RDFS.comment, DCTERMS.description],
                )


        class MyOMDataset(OMDataset):
            track = "Custom"
            ontology_name = "Source-Target"
            source_ontology = MyOntology()
            target_ontology = MyOntology()


    Custom parsers are wired into an ``OMDataset`` in the same way as shown in the
    ``Usage`` section above. Only the parser class changes.

    .. code-block:: python

        task = MyOMDataset()

        dataset = task.collect(
            source_ontology_path="source.owl",
            target_ontology_path="target.owl",
            reference_matching_path="reference.xml",
        )

    This example uses exact language matching for clarity. For broader matching, such as
    ``en`` matching ``en-US``, add a small helper method before selecting the label.

    .. Note:: Related Examples:
                `OAEI parser implementations <https://github.com/sciknoworg/OntoAligner/tree/dev/ontoaligner/ontology/oaei>`_.


.. tab:: ➕ Add New Fields


    Sometimes changing the source of an existing field is not enough. A downstream encoder
    or aligner may need information that is not part of the default parsed concept
    dictionary, such as a normalized label, URI fragment, ontology host, or reusable text
    field.

    In that case, override ``get_class_info``. Call ``super().get_class_info(...)`` first,
    then attach additional values under a dedicated key.

    .. code-block:: python

        from typing import Any
        from urllib.parse import urlparse

        from rdflib import URIRef

        from ontoaligner.ontology import GenericOntology


        class MyMetadataOntology(GenericOntology):
            """
            Extends GenericOntology with extra metadata while keeping the standard
            OntoAligner fields unchanged.
            """

            def get_host(self, iri: str) -> str:
                host = urlparse(iri).hostname
                if host is None:
                    return ""
                return host

            def normalize_text(self, text: str) -> str:
                return " ".join(str(text).lower().strip().split())

            def get_class_info(self, owl_class: URIRef) -> Any:
                class_info = super().get_class_info(owl_class)
                if class_info is None:
                    return None

                class_info["custom"] = {
                    "normalized_label": self.normalize_text(class_info["label"]),
                    "host": self.get_host(class_info["iri"]),
                }

                return class_info


    The standard fields remain available for existing components, while custom components
    can read the extra metadata from ``class_info["custom"]``. This parser can be wrapped
    in an ``OMDataset`` using the same pattern shown above.

    A parser should extract reusable ontology information. An encoder should decide how
    that information is converted into model input text, vectors, prompts, or other
    matching features. If only one encoder needs a specific formatted string, keep that
    formatting in the encoder. If several components need the same extracted value,
    compute it once in the parser.

    These examples show two cases where additional parser fields can be useful. They are
    not required for every custom parser; use them only when the downstream pipeline needs
    the extra metadata.

    .. tab:: 📝 Shared enriched text

        This pattern combines parsed fields such as ``label``,
        ``synonyms``, and ``comment`` into one reusable text representation. It belongs in a
        parser only when the same enriched text should be reused by several downstream
        components.

        .. code-block:: python

            from typing import Any, List

            from rdflib import URIRef

            from ontoaligner.ontology import GenericOntology


            class EnrichedTextOntology(GenericOntology):
                """
                Adds a reusable enriched text field to each parsed concept.
                """

                def _get_synonym_labels(self, synonyms: List[Any]) -> List[str]:
                    labels = []

                    for synonym in synonyms:
                        if isinstance(synonym, dict):
                            label = synonym.get("label", "")
                        else:
                            label = str(synonym)

                        label = label.strip()
                        if label:
                            labels.append(label)

                    return labels

                def get_class_info(self, owl_class: URIRef) -> Any:
                    class_info = super().get_class_info(owl_class)
                    if class_info is None:
                        return None

                    text_parts = [class_info["label"]]
                    synonyms = self._get_synonym_labels(
                        class_info.get("synonyms", [])
                    )

                    if synonyms:
                        text_parts.append("Synonyms: " + ", ".join(synonyms[:5]))

                    comments = class_info.get("comment", [])
                    if comments:
                        text_parts.append("Definition: " + " ".join(comments))

                    class_info["custom"] = {
                        "enriched_text": ". ".join(text_parts).lower().strip()
                    }

                    return class_info

        If the enriched string is needed by one encoder only, keep the formatting logic in
        that encoder.

        .. note::

            For a fuller example of configurable enriched text, see
            `examples/enriched_encoder_generic.py <https://github.com/sciknoworg/OntoAligner/tree/dev/examples/enriched_encoder_generic.py>`__.

    .. tab:: 🧠 OLaLa-style metadata

        Some aligners need richer RDF-level information than the
        default parser output. An OLaLa-style parser keeps the default OntoAligner fields and
        adds a separate namespace containing additional text and URI metadata.

        The important design pattern is not specific to OLaLa. A parser can preserve
        additional fields such as direct text literals, annotation-property text, URI
        fragments, hosts, and normalized labels under a dedicated key. Later stages can then
        reuse those values without re-reading the RDF graph.

        .. note::

            For the complete implementation, see the
            `OLaLaOntology implementation <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/ontology/generic.py#L198-L768>`__.

Before running a full alignment pipeline, parse a small ontology and inspect a few parsed items.

    .. code-block:: python

        task = MyOMDataset()

        dataset = task.collect(
            source_ontology_path="source.owl",
            target_ontology_path="target.owl",
        )

        print(dataset["source"][0].keys())
        print(dataset["source"][0]["label"])
        print(dataset["source"][0].get("custom", {}))


Check that ``iri`` and ``label`` are present, hierarchy fields use the expected dictionary structure, and custom fields are stored under the expected namespace.


.. hint::

    Existing OAEI parsers are useful references for small parser customizations. The
    OLaLa parser is a useful reference for a larger parser extension that preserves
    additional RDF-derived metadata. For more details, see
    `Package Reference > Parsers <../package_reference/parsers.html>`_.


OAEI Parsers
-------------------------

The OAEI tasks (not all of them) datasets are supported within the OntoAligner from the `LLMs4OM: Matching Ontologies with Large Language Models <https://link.springer.com/chapter/10.1007/978-3-031-78952-6_3>`__ empirical study.

The OntoAligner contains several Python modules that include support for the following tracks:

- `Anatomy <https://oaei.ontologymatching.org/2023/anatomy/index.html>`__: Ontology alignments in the anatomical domain.
- `Biodiv <https://oaei.ontologymatching.org/2023/biodiv/index.html>`__: Ontology alignments in the biodiversity domain.
- `BioML <https://krr-oxford.github.io/OAEI-Bio-ML/>`__: Ontology alignments in the biomedical domain, specifically designed for machine learning approaches with train/test sets.
- `CommonKG <https://oaei.ontologymatching.org/2022/commonKG/index.html>`__: Ontology alignments in the common knowledge graph domain.
- `Food <https://oaei.ontologymatching.org/2023/food/index.html>`__: Ontology alignments in the food domain.
- `MSE <https://github.com/EngyNasr/MSE-Benchmark>`__: Ontology alignments in the materials science and engineering domain.
- `Phenotype <https://oaei.ontologymatching.org/2019/>`__: Ontology alignments in the phenotype domain.


The following example demonstrates how to load the ``MaterialInformation-MatOnto`` task from the ``oaei`` track list:

.. code-block:: python

    from ontoaligner.ontology.oaei import MaterialInformationMatOntoOMDataset

    task = MaterialInformationMatOntoOMDataset()

    dataset = task.collect(
        source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="../assets/MI-MatOnto/matchings.xml"
    )

For a simpler import, use:

.. code-block:: python

    from ontoaligner.ontology import MaterialInformationMatOntoOMDataset


.. note::

    Consider reading the following section next for more details on available OAEI parsers.

    * `Package Reference > Parsers <../package_reference/parsers.html>`_

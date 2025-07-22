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
    If you dont specify the ``reference_matching_path``, in the ``OMDataset`` it will be assumed as a empty list ``[]``.


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
    task =  GenericOMDataset()
    task.track = "Geographical"   # optional
    task.ontology_name = "GEO-GeoNames"  # optional
    dataset = task.collect(source_ontology_path="geo.owl", target_ontology_path="geonames.owl")


OAEI Parsers
-------------------------

The OAEI tasks (not all of them) datasets are supported within the OntoAligner from the `LLMs4OM: Matching Ontologies with Large Language Models <https://link.springer.com/chapter/10.1007/978-3-031-78952-6_3>`__ empirical study.

The OntoAligner contains several Python modules that supports the following tracks.

- `Anatomy <https://oaei.ontologymatching.org/2023/anatomy/index.html>`__: Ontology alignments in anatomical domains.
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

    Consider reading the following section next for more details on list of possible OAEI Parsers.

    * `Package Reference > Parsers <../package_reference/parsers.html>`_

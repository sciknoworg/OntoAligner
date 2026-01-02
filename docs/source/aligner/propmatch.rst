PropMatch
=======================

.. sidebar:: **NLTK Data:** PropMatch requires NLTK's POS tagger:

	.. code-block:: python

		import nltk
		nltk.download('averaged_perceptron_tagger')
		nltk.download('averaged_perceptron_tagger_eng')
	::



**PropMatch: is a Property-Based Ontology Matching**. It is a sophisticated property-based ontology matching system that aligns properties between two OWL/RDF ontologies by comparing their labels, domains, and ranges. It implements an iterative refinement algorithm with confidence boosting and supports multiple preprocessing strategies for inferring missing domain/range declarations. The following diagram shows the architecture of PropMatch.


.. hint::

	Key Features

	- **Multi-Component Similarity**: Combines label, domain, and range similarity for robust matching
	- **Iterative Refinement**: Uses aligned classes to boost confidence of related properties
	- **Preprocessing Strategies**: Infers missing domain/range from actual property usage
	- **Flexible Configuration**: Customizable thresholds, similarity weights, and matching parameters
	- **Inverse Property Handling**: Automatically aligns inverse properties when detected
	- **Modular Architecture**: Separate parser, encoder, and aligner components for maximum flexibility

.. note:: PropMatch follows a three-stage pipeline:

	**1. Parser** -- (``PropertyOMDataset`` + ``OntologyProperty``): Loads RDF/OWL ontologies and extracts property information with optional preprocessing.

	**2. Encoder** -- ``PropMatchEncoder```: Reformats parsed property data into a structure suitable for the alignment algorithm.

	**3. Aligner** -- ``PropMatchAligner``: Performs the actual matching using TF-IDF models and iterative refinement.



.. note::

    **Reference:** Sousa, Guilherme, Rinaldo Lima, and Cássia Trojahn. "Results of PropMatch in OAEI 2023." In OM@ ISWC, pp. 178-183. 2023. `https://ceur-ws.org/Vol-3591/oaei23_paper8.pdf <https://ceur-ws.org/Vol-3591/oaei23_paper8.pdf>`_

Usage
-----------

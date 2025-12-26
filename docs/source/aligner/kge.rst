Knowledge Graph Embedding
================================

Graph Embeddings
---------------------------------

.. sidebar:: **Reference:**

    `OntoAligner Meets Knowledge Graph Embedding Aligners <https://arxiv.org/abs/2509.26417>`_


Ontology alignment involves finding correspondences between entities in different ontologies. OntoAligner addresses this challenge by leveraging **Knowledge Graph Embedding (KGE)** models. The core idea of KGE is to represent entities (like classes, properties, individuals) and relations within an ontology as **low-dimensional vectors** in a continuous vector space. These numerical representations (embeddings) are learned to preserve semantic relationships from the original ontology geometrically in the embedding space.

.. hint::

    **Why KGE for Alignment?**

    1) *Semantic Preservation*: KGE models aim to capture the meaning and relationships of entities in their vector representations.
    2) *Scalability*: Working with numerical vectors can be more efficient for large-scale comparison than symbolic matching.
    3) *Similarity Measurement*: Once entities are embedded, their semantic similarity can be easily measured (e.g., using cosine similarity).


OntoAligner's KGE-based alignment process involves several key components that work in sequence. These components are described in the following figure within ``GraphEmbeddingsAligner``.

.. raw:: html

    <div align="center">
     <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/refs/heads/dev/docs/source/img/kge.jpg" width="80%"/>
    </div>


Usage
------------

.. sidebar::

    A usage example is available at `OntoAligner Repository. <https://github.com/sciknoworg/OntoAligner/blob/main/examples/kge.py>`_

This module guides you through a step-by-step process for performing ontology alignment using a KGEs and the OntoAligner library. By the end, you’ll understand how to preprocess data, encode ontologies, generate alignments, evaluate results, and save the outputs in XML and JSON formats.


.. tab:: ➡️ 1: Parser

    The first step is to prepare the ontology data for the KGE model. The **Parser** transforms raw ontology information into a structured format suitable for KGE models.

    .. code-block:: python

        from ontoaligner.ontology import GraphTripleOMDataset

        task  = GraphTripleOMDataset()
        task.ontology_name = "Mouse-Human"
        print("task:", task)
        # >>> task: Track: GraphTriple, Source-Target sets: Mouse-Human

        dataset = task.collect(
            source_ontology_path="assets/mouse-human/source.xml",
            target_ontology_path="assets/mouse-human/target.xml",
            reference_matching_path="assets/mouse-human/reference.xml"
        )
        print("dataset key-values:", dataset.keys())
        # >>> dataset key-values: dict_keys(['dataset-info', 'source', 'target', 'reference'])

        print("Sample source ontology:", dataset['source'][0])

    This will result in the sample source ontology with following metadata:

    .. code-block:: javascript

        [
            {
                'subject': ('http://mouse.owl#MA_0000143', 'tonsil'),
                'predicate': ('http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'type'),
                'object': ('http://www.w3.org/2002/07/owl#Class', 'Class'),
                'subject_is_class': True,
                'object_is_class': False
            },
            ...
        ]
    ::

.. tab:: ➡️ 2: Encoder

    Once the soruce and target ontologies are parsed, the ``GraphTripleEncoder`` creates a triplet representations. The triplet representation is in ``[(Subject Label, Predicate Label, Object Label), ... ]`` format, which is standard input for KGE models.

    .. code-block:: python

        from ontoaligner.encoder import GraphTripleEncoder

        encoder = GraphTripleEncoder()
        encoded_dataset = encoder(**dataset)
    ::

.. tab:: ➡️ 3: Aligner


    After triplets are generated, they are fed into the KGE model. This is the core engine that learns low-dimensional embeddings for all entities and relations present in the triplets. Here lets use ``CovEAligner``, it is a specific implementation of the KGE-based aligner (specifically `ConvE <https://aaai.org/papers/11573-convolutional-2d-knowledge-graph-embeddings/>`_) within the OntoAligner library. It encapsulates the entire process from data ingestion and embedding learning to alignment prediction.

    .. code-block:: python

        from ontoaligner.aligner import ConvEAligner

        kge_params = {
            'device': 'cpu',                  # str: Device to use for training ('cpu' or 'cuda')
            'embedding_dim': 300,             # int: Dimensionality of learned embeddings
            'num_epochs': 50,                 # int: Number of training epochs
            'train_batch_size': 128,          # int: Number of positive triplets per training batch
            'eval_batch_size': 64,            # int: Number of triplets per evaluation batch
            'num_negs_per_pos': 5,            # int: Number of negative samples per positive triplet
            'random_seed': 42,                # int: Seed for reproducibility
        }

        aligner = ConvEAligner(**kge_params)

        matchings = aligner.generate(input_data=encoded_dataset)

    .. note::

        The ``.generate`` function will do the training and then matching.

    ::

.. tab:: ➡️ 4: Post-Process

    This step focuses on post-processing predicted matchings, potentially utilizing a similarity score for filtering and applying cardinality based processing, and subsequently evaluating their quality against a reference dataset to assess performance before and after post-processing.

    .. code-block:: python

        from ontoaligner.postprocess import graph_postprocessor

        processed_matchings = graph_postprocessor(predicts=matchings, threshold=0.5)

    ::

.. tab:: ➡️ 5:  Evaluate and Export

    The following code will compare the generated alignments with reference matchings. Then save the matchings in both XML and JSON formats for further analysis or use. Feel free to use any of the techniques.

    .. code-block:: python

        from ontoaligner.utils import metrics

        evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
        print("Matching Evaluation Report:\n", evaluation)

        evaluation = metrics.evaluation_report(predicts=processed_matchings, references=dataset['reference'])
        print("Matching Evaluation Report -- after post-processing:\n", evaluation)


    .. tab:: 📄 <> Export matchings to XML

        ::

            from ontoaligner.utils import metrics

            xml_str = xmlify.xml_alignment_generator(matchings=processed_matchings)
            with open("matchings.xml", "w", encoding="utf-8") as xml_file:
                xml_file.write(xml_str)

    .. tab::  # 🧾 {} Export matchings to JSON

        ::

            with open("matchings.json", "w", encoding="utf-8") as json_file:
                json.dump(processed_matchings, json_file, indent=4, ensure_ascii=False)
    ::







KGE Aligners
----------------------



The ``ontoaligner.aligner.graph`` module provides a suite of graph embedding-based aligners built on top of popular KGE models. These aligners leverage link prediction objectives and low-dimensional vector spaces to learn semantic representations of entities, facilitating accurate ontology alignment even across heterogeneous structures. Each aligner wraps a specific KGE model implemented through the PyKEEN framework, allowing plug-and-play integration and consistent similarity scoring across models. Some models include custom similarity functions to better capture semantic distance in complex embedding spaces (e.g., complex numbers or quaternions).

The following table lists the available KGE aligners:

.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - Aligner Name
     - Description
     - Link

   * - ``ConvEAligner``
     - Based on ConvE, which uses 2D convolutions over reshaped entity and relation embeddings to model complex interactions.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L17-L18>`_
   * - ``TransDAligner``
     - Based on TransD, which constructs relation-specific projection matrices dynamically from both entity and relation vectors.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L21-L22>`_
   * - ``TransEAligner``
     - Based on TransE, a translation-based model that learns embeddings where :math:`h + r \approx t`.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L25-L26>`_
   * - ``TransFAligner``
     - Based on TransF, which enables flexible translations for complex relations without increasing model complexity.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L29-L230>`_
   * - ``TransHAligner``
     - Based on TransH, which projects entities onto relation-specific hyperplanes before translation.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L33-L234>`_
   * - ``TransRAligner``
     - Based on TransR, which embeds entities and relations in separate spaces using relation-specific projections.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L37-L38>`_
   * - ``DistMultAligner``
     - Based on DistMult, a bilinear model that uses diagonal matrices for efficient relational modeling.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L41-L42>`_
   * - ``ComplExAligner``
     - Based on ComplEx, which uses complex-valued embeddings to model symmetric and antisymmetric relations; includes a custom similarity function using real parts of complex dot products.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L45-L49>`_
   * - ``HolEAligner``
     - Based on HolE, which combines compositional and holographic representations using circular correlation.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L51-L52>`_
   * - ``RotatEAligner``
     - Based on RotatE, which models relations as rotations in complex space and supports rich relational patterns; includes a similarity override.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L55-L60>`_
   * - ``SimplEAligner``
     - Based on SimplE, which learns dependent embeddings for each entity and supports fully expressive factorization.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L62-L63>`_
   * - ``CrossEAligner``
     - Based on CrossE, which learns both general and triple-specific embeddings to capture bidirectional interactions.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L66-L67>`_
   * - ``BoxEAligner``
     - Based on BoxE, which models relations as boxes in vector space to support hierarchies and logical rules.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L70-L71>`_
   * - ``CompGCNAligner``
     - Based on CompGCN, a graph convolutional network designed for multi-relational graphs using composition operations.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L74-L75>`_
   * - ``MuREAligner``
     - Based on MuRE, which embeds entities in hyperbolic space to better model hierarchies and relation-specific transformations.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L78-L79>`_
   * - ``QuatEAligner``
     - Based on QuatE, which uses quaternion embeddings and custom similarity logic to model expressive 4D rotations and relational structure.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L82-L133>`_
   * - ``SEAligner``
     - Based on SE, a neural model that embeds symbolic knowledge into vector space using learned neural transformations.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/kge/models.py#L134-L135>`_

To use KGE aligner based technique:

.. code-block:: python

        from ontoaligner.aligner import TransEAligner

        aligner = TransEAligner()

        matchings = aligner.generate(input_data=...)

If the desired model is not avaliable in OntoAligner, then:

.. code-block:: python

    from ontoaligner.aligner.graph import GraphEmbeddingAligner

    class CustomKGEAligner(GraphEmbeddingAligner):
        model = "RESCAL"

    aligner = CustomKGEAligner()
    matchings = aligner.generate(input_data=...)


Here ``RESCAL`` is our custom KGE model.

.. note::

    For possible models please take a look at `PyKEEN > Models <https://pykeen.readthedocs.io/en/latest/reference/models.html#classes>`_.

KGE Retriever
----------------------

.. sidebar:: Key Parameters:

		- ``retruever``: boolean
		- ``top_K``: integer

In addition to one-to-one alignments, OntoAligner also supports retriever-based alignment. When retriever mode is enabled (``retriever=True``), the aligner returns the top-k candidate target entities for each source entity, along with their similarity scores (similar to retriever aligner). This model is useful if you want to build downstream candidate filtering pipelines, apply human-in-the-loop validation, or integrate with reranking modules (e.g., LLMs or supervised classifiers).

Here is the example on how to use KGE Aligner as a retriever model:

.. code-block:: python

    from ontoaligner.aligner import TransEAligner

    # Enable retriever mode and request top-3 candidates per source entity
    aligner = TransEAligner(retriever=True, top_k=3)

    matchings = aligner.generate(input_data=encoded_dataset)

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Description

   * - **KGE Default mode**
     - In KGE aligners, the default mode is ``retriever=False``, where it produces **one-to-one** alignments, where each source entity is matched to the single most similar target entity.
   * - **KGE Retriever mode**
     - In KGE aligners, the default mode is ``retriever=True``, where it produces **one-to-many** alignments, where each source entity is matched to multiple target entities. Example output:



.. tab:: ➡️ KGE Retriever Mode Example output

	::

		[
		   {
		     "source": "http://mouse.owl#MA_0000143",
		     "target-cands": [
		         "http://human.owl#HBA_0000214",
		         "http://human.owl#HBA_0000762",
		         "http://human.owl#HBA_0000891"
		     ],
		     "score-cands": [0.87, 0.82, 0.77]
		   },
		   ...
		]



.. tab:: ➡️ KGE Default Mode Example output

	::

		{
		    'source': 'http://mouse.owl#MA_0000143',
		    'target': 'http://human.owl#HBA_0000214',
		    'score': 0.87
		}


.. note::

    Consider reading the following section next:

    * `Package Reference > Aligners <../package_reference/aligners.html>`_

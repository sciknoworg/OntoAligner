Overview
===========


.. raw:: html

   <div align="center">
     <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/refs/heads/dev/docs/source/img/ontoaligner-pip.jpg" alt="OntoAligner Overview" width="840px"/>
   </div>

OntoAligner is a modular, extensible, and efficient framework for ontology alignment that integrates classical heuristics, retrieval-based methods, and large language models (LLMs). It is designed to support a wide range of ontology alignment (OA) scenarios—from lightweight matching to advanced semantic reasoning—with built-in support for evaluation and export.


.. tab:: 🧩 Parser



    The ``Parser`` module serves as the entry point of OntoAligner, handling ontology ingestion and alignment data loading. Key components include:

    1) ``OntologyParser`` that supports loading ontologies and extracts class/property names, IRIs, hierarchies, synonyms, annotations, and any relevant informations.

    2) ``AlignmentsParser`` for loading ground truth alignments used for evaluation, however in a case of specific dataset doesnt support the references this module need to be ignored.

    .. note::

        🔧 Checkout parsers at `Developer Guide > Parsers <../developerguide/parsers.html>`_
    ::


.. tab:: 🔠 Encoder

    After parsing, the ``Encoder`` module transforms ontological concepts into structured representations suited for similarity estimation or prompt-based inference given the nature of aligner model. Here are few list of supported encoders:

    1. **Lightweight Encoder**: Combines concept metadata (Concept, Children, Parent) into simple natural language strings for heuristic-based aligners.

    2. **LLM Encoder**: Prepares structured prompt-compatible inputs for language models, extracting and formatting parent-child relationships per concept.

    3. **RAG Encoder**: Tailored for Retrieval-Augmented Generation (RAG)-based alignment using in-context vectors (ICV) to support few-shot and in-context learning setups.

    Each encoder is pluggable and configurable, allowing flexible adaptation to different use cases.

    ::


.. tab:: 🧠 Aligners

    The Aligner Module is the core of OntoAligner and is responsible for discovering mappings between entities in two ontologies. It includes a diverse suite of alignment algorithms grouped into different categories: ``Lightweight``, ``Retrieval``, ``LLM``, ``RAG``, etc.

    .. list-table::
       :header-rows: 1
       :widths: 20 80

       * - Aligner
         - Description

       * - **Lightweight Aligners**
         - Fast, heuristic-based methods using fuzzy matching and token similarity with configurable thresholds.
           Suitable for large-scale, low-complexity tasks.

       * - **Retrieval Aligners**
         - Machine learning-based methods using vector representations: ``TFIDF``, ``SVM``, and ``Sentence-Transformers``.
           Designed for semantic similarity retrieval across mid-to-large ontologies.

       * - **LLM Aligners**
         - Uses pretrained LLMs from HuggingFace to evaluate alignment based on natural language prompts.
           Offers strong semantic alignment for small-scale ontologies (≤200 concepts) due to quadratic complexity.

       * - **RAG Aligners**
         - A hybrid approach that enhances accuracy by combining retrieval and LLM reasoning.
           Retrieves contextual information.
           Computes alignment decisions using logit-based scoring instead of generation.
           Reduces GPU usage with single forward-pass inference.

    ::


.. tab:: 🔄 Post-Processing

    Fully modular and extendable to accommodate custom post-alignment techniques.

    Mapper: Maps generated texts (e.g., LLM outputs) to ontology classes.

    Filtering: Applies heuristic and rule-based strategies to ensure consistency and precision.

    ::


.. tab:: 📊 Evaluation and Exporters

    📈 Evaluator: Computes standard OA metrics: Precision, Recall, and F1-score. Supports comparison across different aligners and configurations.

    📤 Exporter: Supports alignment export in common formats: XML and JSON.

    ::



Usage
-----------

.. tab:: 🛠️  Installation


    You can install **OntoAligner** using pip:

    .. code-block::

        pip install -U OntoAligner

    See `installation <installation.html>`_ for further installation options.

    ::


.. tab:: 🚀  Quickstart

    See the `Quickstart <quickstart.html>`_ for more quick information on how to use OntoAligner.

    ::


Working with OntoAligner is straightforward:

.. code-block:: python

    from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
    from ontoaligner.utils import metrics, xmlify
    from ontoaligner.aligner import MistralLLMBERTRetrieverRAG
    from ontoaligner.encoder import ConceptParentRAGEncoder
    from ontoaligner.postprocess import rag_hybrid_postprocessor

    # Step 1: Initialize the dataset object for MaterialInformation MatOnto dataset
    task = MaterialInformationMatOntoOMDataset()
    print("Test Task:", task)

    # Step 2: Load source and target ontologies along with reference matchings
    dataset = task.collect(
        source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="assets/MI-MatOnto/matchings.xml"
    )

    # Step 3: Encode the source and target ontologies
    encoder_model = ConceptParentRAGEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

    # Step 4: Define configuration for retriever and LLM
    retriever_config = {"device": 'cuda', "top_k": 5,}
    llm_config = {"device": "cuda", "max_length": 300, "max_new_tokens": 10, "batch_size": 15}

    # Step 5: Initialize Generate predictions using RAG-based ontology matcher
    model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
    model.load(llm_path = "mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")
    predicts = model.generate(input_data=encoded_ontology)

    # Step 6: Apply hybrid postprocessing
    hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(predicts=predicts,
                                                                ir_score_threshold=0.1,
                                                                llm_confidence_th=0.8)

    evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
    print("Hybrid Matching Evaluation Report:", evaluation)

    # Step 7: Convert matchings to XML format and save the XML representation
    xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)
    open("matchings.xml", "w", encoding="utf-8").write(xml_str)

What is Next?
----------------

**Quickstart**:

* How to dive quickly into ontology alignments? `Getting Started > Quickstart <quickstart.html>`_.

**Developer Guide**:

* How to parse ontologies for ontology aligment? `Developer Guide > Parsers <../developerguide/parsers.html>`_.

**Aligner Techniques**:

* How to use *Lightweight* Aligner? `Aligners > Lightweight <../aligner/lightweight.html>`_
* How to use *Retrieval* Aligner? `Aligners > Retrieval <../aligner/retriever.html>`_
* How to use *Large Language Model* Aligner? `Aligners > Large Language Model <../aligner/llm.html>`_
* How to use *Retrieval Augmented Generation* Aligner? `Aligners > Retrieval Augmented Generation <../aligner/rag.html>`_

**Use Casses**:

* How OntoAligner can be used in e-Commerce? `Use Cases > eCommerce <../usecases/ecommerce.html>`_

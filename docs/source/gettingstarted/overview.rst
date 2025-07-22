Overview
=========================


.. raw:: html

   <div align="center">
     <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/refs/heads/dev/docs/source/img/ontoaligner-pip.jpg" alt="OntoAligner Overview" width="840px"/>
   </div>

OntoAligner is a modular, extensible, and efficient framework for ontology alignment that integrates classical heuristics, retrieval-based methods, and large language models (LLMs). It is designed to support a wide range of ontology alignment (OA) scenarios—from lightweight matching to advanced semantic reasoning—with built-in support for evaluation and export.

Parser
-------------------
The ``Parser`` module serves as the entry point of OntoAligner, handling ontology ingestion and alignment data loading.

Key components include:
1) ``OntologyParser`` that supports loading ontologies and extracts class/property names, IRIs, hierarchies, synonyms, annotations, and any relevant informations.
2) ``AlignmentsParser`` for loading ground truth alignments used for evaluation, however in a case of specific dataset doesnt support the references this module need to be ignored.

.. note:: 🔧

    See `



Encoder
------------
After parsing, the `Encoder` module transforms ontological concepts into structured representations suited for similarity estimation or prompt-based inference given the nature of aligner model.

Supported Encoders:
Lightweight Encoder
Combines concept metadata (Concept, Children, Parent) into simple natural language strings for heuristic-based aligners.

LLM Encoder
Prepares structured prompt-compatible inputs for language models, extracting and formatting parent-child relationships per concept.

RAG Encoder
Tailored for Retrieval-Augmented Generation (RAG)-based alignment using in-context vectors (ICV) to support few-shot and in-context learning setups.

Each encoder is pluggable and configurable, allowing flexible adaptation to different use cases.





Aligners
----------

The Aligner Module is the core of OntoAligner and is responsible for discovering mappings between entities in two ontologies. It includes a diverse suite of alignment algorithms grouped into different categories: ``Lightweight``, ``Retrieval``, ``LLM``, ``RAG``, etc.

🧠 Aligner Module
The Aligner Module implements the core ontology matching algorithms, divided into four categories:

1. Lightweight Aligners
Fast, heuristic-based methods using fuzzy matching and token similarity with configurable thresholds. Suitable for large-scale, low-complexity tasks.

2. Retrieval Aligners
Machine learning-based methods using vector representations:

TFIDF, SVM, and Sentence-Transformers from Hugging Face.

Designed for semantic similarity retrieval across mid-to-large ontologies.

3. LLM Aligners
Uses pretrained LLMs from Hugging Face to evaluate alignment based on natural language prompts. Offers strong semantic alignment for small-scale ontologies (≤200 concepts) due to quadratic complexity.

4. RAG Aligners
A hybrid approach that enhances accuracy by combining retrieval and LLM reasoning:

Retrieves contextual information.

Computes alignment decisions using logit-based scoring instead of generation.

Reduces GPU usage with single forward-pass inference.

🔄 Post-Processing:
Mapper: Maps generated texts (e.g., LLM outputs) to ontology classes.

Filtering: Applies heuristic and rule-based strategies to ensure consistency and precision.

Fully modular and extendable to accommodate custom post-alignment techniques.



Evaluation and Exporters
-----------------------------
📈 Evaluator:
Computes standard OA metrics: Precision, Recall, and F1-score.

Supports comparison across different aligners and configurations.

📤 Exporter:
Supports alignment export in common formats: XML and JSON.


🔧 Design Philosophy and Features
OntoAligner is engineered for:

Extensibility: Easily add custom alignment algorithms via API (F11, F12).

Modularity: Each component is self-contained and independently configurable.

Scalability: Optimized for large ontologies (thousands of entities) with performance-conscious design.

Usability: Clear documentation, code comments, and tutorials for all levels of users.

| Feature                     | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| 🧬 OWL & RDF Support        | Parses standard ontology formats (local or remote)    |
| 🔄 Modular Encoders         | Concept-level, prompt-based, and RAG-specific formats |
| 🤖 Multi-Strategy Aligners  | Heuristics, ML, LLMs, and RAG-based matching          |
| 🧹 Post-Processing Pipeline | Rule-based and LLM-based refinement modules           |
| 📈 Metrics & Comparison     | Built-in evaluator with multiple OA metrics           |
| 📦 Export Support           | JSON and XML alignment export                         |


List of Aligners
----------------------------

The table below is structured with the following columns:

- **Category**: The type or category of the aligners.

- **Models with Links**: A list of models belonging to the category.

.. note::
   Use the links to access the implementation details for each model. This catalog helps quickly locate and explore the source code for different models in the **OntoAligner** repository.


.. list-table::
   :header-rows: 1
   :class: catalog-table

   * - **Category**
     - **Models with Links**
   * - **Lightweight Aligners**
     - `SimpleFuzzySMLightweight <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/lightweight/models.py#L23-L47>`__, `WeightedFuzzySMLightweight <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/lightweight/models.py#L50-L74>`__, `TokenSetFuzzySMLightweight <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/lightweight/models.py#L77-L101>`__
   * - **Retrieval Aligners**
     - `AdaRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L191-L250>`__, `BM25Retrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L109-L172>`__, `SBERTRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L28-L42>`__, `SVMBERTRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L175-L188>`__, `TFIDFRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L45-L106>`__
   * - **LLM Aligners**
     - `AutoModelDecoderLLM <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/llm/models.py#L31-L46>`__, `FlanT5LEncoderDecoderLM <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/llm/models.py#L13-L28>`__, `GPTOpenAILLM <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/llm/models.py#L49-L61>`__
   * - **RAG Aligners**
     - `FalconLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L127-L143>`__, `FalconLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L146-L162>`__, `GPTOpenAILLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L89-L105>`__, `GPTOpenAILLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L108-L124>`__, `LLaMALLMAdaRetrieverRAG <https://ontoaligner.readthedocs.io/package_reference/ontolog_matchers.html#module-ontoaligner.aligner.rag.models>`__, `LLaMALLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L32-L48>`__, `MPTLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L203-L219>`__, `MPTLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L222-L238>`__, `MambaLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L241-L257>`__, `MambaLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L260-L276>`__, `MistralLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L51-L67>`__, `MistralLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L70-L86>`__, `VicunaLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L165-L181>`__, `VicunaLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/rag/models.py#L184-L200>`__
   * - **FewShot-RAG Aligners**
     - `FalconLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L105-L117>`__, `FalconLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L120-L132>`__, `GPTOpenAILLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L75-L87>`__, `GPTOpenAILLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L90-L102>`__, `LLaMALLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L15-L27>`__, `LLaMALLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L30-L42>`__, `MPTLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L165-L177>`__, `MPTLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L180-L192>`__, `MambaLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L195-L207>`__, `MambaLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L210-L222>`__, `MistralLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L45-L57>`__, `MistralLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L60-L72>`__, `VicunaLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L135-L147>`__, `VicunaLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/fewshot/models.py#L150-L162>`__
   * - **ICV-RAG Aligners**
     - `FalconLLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L53-L69>`__, `FalconLLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L72-L88>`__, `LLaMALLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L15-L31>`__, `LLaMALLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L34-L50>`__, `MPTLLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L129-L145>`__, `MPTLLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L148-L164>`__, `VicunaLLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L91-L107>`__, `VicunaLLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/icv/models.py#L110-L126>`__

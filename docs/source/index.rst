> **The vision is to create a unified hub that brings together a wide range of ontology alignment models, making integration seamless for researchers and practitioners.**

OntoAligner Documentation
===========================

Ontology Alignment (OA) is fundamental for achieving semantic interoperability across diverse knowledge systems. **OntoAligner** (a.k.a Ontology Aligner), a modular Python toolkit for ontology alignment, designed to address current limitations with existing tools faced by practitioners. Existing tools are limited in scalability, modularity, and ease of integration with recent AI advances. OntoAligner provides a flexible architecture integrating existing lightweight OA techniques such as fuzzy matching but goes beyond by supporting contemporary methods with retrieval-augmented generation and large language models for OA. The current framework prioritizes extensibility, enabling researchers to integrate custom alignment algorithms and datasets. With OntoAligner you can handle large-scale ontologies efficiently with few lines of code while delivering high alignment quality. By making OntoAligner open-source, we aim to provide a resource that fosters innovation and collaboration within the OA community, empowering researchers and practitioners with a toolkit for reproducible OA research and real-world applications.

OntoAligner was created by `Scientific Knowledge Organization (SciKnowOrg group) <https://github.com/sciknoworg/>`_ at `Technische Informationsbibliothek (TIB) <https://www.tib.eu/de/>`_. Don't hesitate to open an issue on the `OntoAligner repository <https://github.com/sciknoworg/OntoAligner>`_ if something is broken or if you have further questions.


.. note::

    OntoAligner was  awarded the `🏆 Best Resource Paper Award at ESWC 2025 <https://2025.eswc-conferences.org/eswc-2025-best-paper-reviewer-awards/>`_

Usage
=====

.. note::

    See the `Quickstart <gettingstarted/quickstart.html>`_ for more quick information on how to use OntoAligner.


.. tip::

    You can install **OntoAligner** using pip:

    .. code-block:: cmd

        pip install -U OntoAligner
    We recommend **Python 3.10+** and `PyTorch 1.4.0+ <https://pytorch.org/get-started/locally/>`_, and `transformers v4.41.0+ <https://github.com/huggingface/transformers>`_. See `installation <gettingstarted/installation.html>`_ for further installation options.


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
===============

**Parsers**:

* How to parse ontologies for ontology aligment? `Getting Started > Ontology Parsers <gettingstarted/parsers.html>`_.

**Aligner Models**:

* How to use *Lightweight* Aligner? `Aligners > Lightweight <aligner/lightweight.html>`_
* How to use *Retrieval* Aligner? `Aligners > Retrieval <aligner/retriever.html>`_
* How to use *Large Language Model* Aligner? `Aligners > Large Language Model <aligner/llm.html>`_
* How to use *Retrieval Augmented Generation* Aligner? `Aligners > Retrieval Augmented Generation <aligner/rag.html>`_

**Use Casses**:

* How OntoAligner can be used in e-Commerce? `Use Cases > eCommerce <usecases/ecommerce.html>`_


Citing
=========

If you find this repository helpful, feel free to cite our publication `OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment <https://link.springer.com/chapter/10.1007/978-3-031-94578-6_10>`_:

 .. code-block:: bibtex

    @inproceedings{babaei2025ontoaligner,
      title={OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment},
      author={Babaei Giglou, Hamed and D’Souza, Jennifer and Karras, Oliver and Auer, S{\"o}ren},
      booktitle={European Semantic Web Conference},
      pages={174--191},
      year={2025},
      organization={Springer}
    }

or our related work `LLMs4OM: Matching Ontologies with Large Language Models <https://link.springer.com/chapter/10.1007/978-3-031-78952-6_3>`_:

 .. code-block:: bibtex

  @inproceedings{babaei2024llms4om,
      title={LLMs4OM: Matching Ontologies with Large Language Models},
      author={Babaei Giglou, Hamed and D’Souza, Jennifer and Engel, Felix and Auer, S{\"o}ren},
      booktitle={European Semantic Web Conference},
      pages={25--35},
      year={2024},
      organization={Springer}
    }


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :glob:
   :hidden:

   gettingstarted/overview
   gettingstarted/installation
   gettingstarted/quickstart
   gettingstarted/parsers


.. toctree::
   :caption: Aligners
   :titlesonly:
   :hidden:
   :maxdepth: 1

   aligner/lightweight
   aligner/retriever
   aligner/llm
   aligner/rag



.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :hidden:

   usecases/ecommerce


.. toctree::
   :maxdepth: 1
   :caption: Package Reference
   :hidden:

   package_reference/pipeline
   package_reference/parsers
   package_reference/encoders
   package_reference/aligners
   package_reference/postprocess

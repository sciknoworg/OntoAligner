

.. raw:: html

   <div align="center">
     <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/main/docs/source/img/logo-ontoaligner.png" alt="OntoLearner Logo" width="700"/>
   </div>


.. raw:: html

    <div style="text-align: center;">
        <a href="https://badge.fury.io/py/OntoAligner">
            <img src="https://badge.fury.io/py/OntoAligner.svg" alt="PyPI version">
        </a>
        <a href="https://static.pepy.tech/badge/ontoaligner">
            <img src="https://static.pepy.tech/badge/ontoaligner" alt="PyPI downloads">
        </a>
        <a href="https://www.apache.org/licenses/LICENSE-2.0">
            <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
        </a>
    </div>


Ontologies are a key building block for many applications, such as database integration, knowledge graphs, e-commerce, semantic web services, or social networks. However, evolving systems within the semantic web generally adopt different ontologies. Hence, ontology alignment, the process of identifying correspondences between entities in different ontologies, is a critical task in knowledge engineering. To this endover, **OntoAligner** is a comprehensive modular and robust Python toolkit for ontology alignment built to make ontology alignment/matching easy to use for everyone.

OntoAligner was created by `Scientific Knowledge Organization (SciKnowOrg group) <https://github.com/sciknoworg/>`_ at `Technische Informationsbibliothek (TIB) <https://www.tib.eu/de/>`_. Don't hesitate to open an issue on the `OntoAligner repository <https://github.com/sciknoworg/OntoAligner>`_ if something is broken or if you have further questions.



.. tab:: Quickstart

    See the `Quickstart <gettingstarted/quickstart.html>`_ for more quick information on how to use OntoAligner.
    ::



.. tab:: Installation

    You can install *ontoaligner* using pip:
    ::

       pip install -U ontoaligner

    See `installation <gettingstarted/installation.html>`_ for further installation options.

Usage
=====

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


Citing
=========

If you find this repository helpful, feel free to cite our publication `OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment <https://arxiv.org/abs/2503.21902>`_:

 .. code-block:: bibtex

    @article{giglou2025ontoaligner,
      title={Ontoaligner: A comprehensive modular and robust python toolkit for ontology alignment},
      author={Giglou, Hamed Babaei and D'Souza, Jennifer and Karras, Oliver and Auer, S{\"o}ren},
      journal={arXiv preprint arXiv:2503.21902},
      year={2025}
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

   gettingstarted/installation
   gettingstarted/quickstart


.. toctree::
   :caption: How to use?
   :hidden:
   :maxdepth: 1

   howtouse/models
   howtouse/parsers


.. toctree::
   :caption: Aligners
   :glob:
   :titlesonly:
   :hidden:
   :maxdepth: 1

   Lightweight <aligner/lightweight>
   Retrieval <aligner/retriever>
   Large Language Models <aligner/llm>
   Retrieval Augmented Generation <aligner/rag>



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
   package_reference/base
   package_reference/ontology
   package_reference/aligner
   package_reference/encoder
   package_reference/postprocess
   package_reference/utils

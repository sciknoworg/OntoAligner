Retrieval Augmented Generation
================================

This tutorial demonstrates how to use the OntoAligner library for ontology matching, leveraging Mistral LLM and RAG techniques. The process involves encoding ontologies, generating matches, and refining them using heuristic and hybrid postprocessing methods.

Step 1. Import Required Modules
********************************************

First, we import the necessary modules from the OntoAligner library and Python’s standard library:

.. code-block:: python

    import json
    from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
    from ontoaligner.utils import metrics, xmlify
    from ontoaligner.ontology_matchers import MistralLLMBERTRetrieverRAG
    from ontoaligner.encoder import ConceptParentRAGEncoder
    from ontoaligner.postprocess import rag_hybrid_postprocessor, rag_heuristic_postprocessor


Step 2. Define the Task and Load Ontologies
********************************************

We use the ``MaterialInformationMatOntoOMDataset`` class to load the source and target ontologies along with the reference matching file:

.. code-block:: python

    task = MaterialInformationMatOntoOMDataset()
    print("Test Task:", task)

    dataset = task.collect(
        source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="../assets/MI-MatOnto/matchings.xml"
    )


Step 3. Encode Ontologies
********************************************

We use the ``ConceptParentRAGEncoder`` to encode the source and target ontologies:

.. code-block:: python

    encoder_model = ConceptParentRAGEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

Step 4. Configure the Retriever and LLM
********************************************

Set up the configurations for the retriever and LLM modules. This example uses a CUDA device and defines thresholds for retrieval and LLM:

.. code-block:: python

    retriever_config = {
        "device": 'cuda',
        "top_k": 5,
        "threshold": 0.1,
    }

    llm_config = {
        "device": "cuda",
        "max_length": 300,
        "max_new_tokens": 10,
        "huggingface_access_token": "",
        "device_map": 'balanced',
        "batch_size": 15,
        "answer_set": {
            "yes": ["yes", "correct", "true", "positive", "valid"],
            "no": ["no", "incorrect", "false", "negative", "invalid"]
        }
    }

Step 5. Generate Predictions
********************************************

Create an instance of ``MistralLLMBERTRetrieverRAG`` and generate predictions:

.. code-block:: python

    model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
    model.load(llm_path = "mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")

    predicts = model.generate(input_data=encoded_ontology)

Step 6. Postprocess Matches
********************************************

*Heuristic Postprocessing*: Automatically determine thresholds for retrieval and LLM confidence using the heuristic method:

.. code-block:: python

    heuristic_matchings, heuristic_configs = rag_heuristic_postprocessor(predicts=predicts, topk_confidence_ratio=3, topk_confidence_score=3)
    evaluation = metrics.evaluation_report(predicts=heuristic_matchings, references=dataset['reference'])
    print("Heuristic Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Heuristic Matching Obtained Configuration:", heuristic_configs)

*Hybrid Postprocessing*: Apply fixed thresholds to filter matchings:

.. code-block:: python

    hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(predicts=predicts, ir_score_threshold=0.1, llm_confidence_th=0.8)
    evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
    print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Hybrid Matching Obtained Configuration:", hybrid_configs)

Step 7. Save Matchings in XML Format
********************************************

Finally, convert the matchings to XML format for compatibility with ontology alignment tools and save them:

.. code-block:: python

    xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

    output_file_path = "matchings.xml"
    with open(output_file_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_str)


Summary
******************

In this tutorial, we demonstrated:

* Loading and encoding ontologies
* Using Mistral LLM with RAG for ontology matching
* Refining results with heuristic and hybrid postprocessing
* Saving results in XML format

You can customize the configurations and thresholds based on your specific dataset and use case. For more details, refer to the :doc:`../package_reference/postprocess`

Fewshot RAG
===============

This tutorial demonstrates a pipeline for ontology alignment using the OntoAligner framework. It involves dataset preparation, encoding ontologies, alignment using a retrieval-augmented generation (RAG) model, hybrid postprocessing of matchings, and evaluation. The final matchings are saved in XML format.

Lets, prepare the source, target, and reference matching files for alignment.

.. code-block:: python

    from ontoaligner.ontology import MaterialInformationMatOntoOMDataset

    task = MaterialInformationMatOntoOMDataset()
    dataset = task.collect(
        source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="../assets/MI-MatOnto/matchings.xml"
    )
    print("Dataset loaded:", dataset)


The second step is to encode the ontology for appropiate format for Few-Shot RAG models Encode the ontologies for further alignment.

.. code-block:: python

    from ontoaligner.encoder import ConceptParentFewShotEncoder

    encoder_model = ConceptParentFewShotEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])
    print("Encoded Ontologies:", encoded_ontology)

Now, use a Fewshot Retrieval-Augmented Generation (RAG) model for ontology alignment.

.. code-block:: python

    from ontoaligner.ontology_matchers import MistralLLMBERTRetrieverFSRAG

    config = {
        "retriever_config": {"device": 'cuda', "top_k": 5, "threshold": 0.1},
        "llm_config": {
            "device": "cuda", "batch_size": 32,
            "answer_set": {"yes": ["yes", "true"], "no": ["no", "false"]}
        }
    }
    model = MistralLLMBERTRetrieverFSRAG(positive_ratio=0.7, n_shots=5, **config)
    model.load(llm_path="mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")

    predicts = model.generate(input_data=encoded_ontology)

And lastly, do the post-processing and evaluate the matchings then eport the results of alignments.

.. code-block:: python

    from ontoaligner.postprocess import rag_hybrid_postprocessor
    from ontoaligner.utils import metrics, xmlify

    hybrid_matchings, _ = rag_hybrid_postprocessor(
        predicts=predicts,
        ir_score_threshold=0.3,
        llm_confidence_th=0.5
    )

    evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
    print("Evaluation Report:", evaluation)

    xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)
    with open("matchings.xml", "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_str)

This workflow demonstrates how to efficiently align ontologies using OntoAligner with minimal setup. Fine-tune parameters like thresholds and retriever configurations to improve performance.

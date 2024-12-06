Retrieval Augmented Generation
================================

This tutorial demonstrates how to use the OntoAligner library for ontology matching, leveraging Mistral LLM and RAG techniques. The process involves encoding ontologies, generating matches, and refining them using heuristic and hybrid postprocessing methods.

Step 1. Import Required Modules
---------------------------------

First, we import the necessary modules from the OntoAligner library and Python’s standard library:

.. code-block:: python

    import json
    from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
    from ontoaligner.utils import metrics, xmlify
    from ontoaligner.ontology_matchers import MistralLLMBERTRetrieverRAG
    from ontoaligner.encoder import ConceptParentRAGEncoder
    from ontoaligner.postprocess import rag_hybrid_postprocessor, rag_heuristic_postprocessor

Step 2. Define the Task and Load Ontologies
---------------------------------------------

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
---------------------------------------------

We use the ``ConceptParentRAGEncoder`` to encode the source and target ontologies:

.. code-block:: python

    encoder_model = ConceptParentRAGEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

Step 4. Configure the Retriever and LLM
---------------------------------------------

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
---------------------------------------------
Create an instance of ``MistralLLMBERTRetrieverRAG`` and generate predictions:

.. code-block:: python

    model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
    predicts = model.generate(input_data=encoded_ontology)

Step 6. Postprocess Matches
---------------------------------------------

**Heuristic Postprocessing**: Automatically determine thresholds for retrieval and LLM confidence using the heuristic method:

.. code-block:: python

    heuristic_matchings, heuristic_configs = rag_heuristic_postprocessor(predicts=predicts, topk_confidence_ratio=3, topk_confidence_score=3)
    evaluation = metrics.evaluation_report(predicts=heuristic_matchings, references=dataset['reference'])
    print("Heuristic Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Heuristic Matching Obtained Configuration:", heuristic_configs)

**Hybrid Postprocessing**: Apply fixed thresholds to filter matchings:

.. code-block:: python

    hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(predicts=predicts, ir_score_threshold=0.1, llm_confidence_th=0.8)
    evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
    print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Hybrid Matching Obtained Configuration:", hybrid_configs)

Step 7. Save Matchings in XML Format
---------------------------------------------
Finally, convert the matchings to XML format for compatibility with ontology alignment tools and save them:

.. code-block:: python

    xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

    output_file_path = "matchings.xml"
    with open(output_file_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_str)


Summary
---------------------------------------------
In this tutorial, we demonstrated:

* Loading and encoding ontologies
* Using Mistral LLM with RAG for ontology matching
* Refining results with heuristic and hybrid postprocessing
* Saving results in XML format

You can customize the configurations and thresholds based on your specific dataset and use case. For more details, refer to the :doc:`package_reference/postprocess`

Full Code
--------------------------
Here is the complete script for reference:

.. code-block:: python

    import json
    from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
    from ontoaligner.utils import metrics, xmlify
    from ontoaligner.ontology_matchers import MistralLLMBERTRetrieverRAG
    from ontoaligner.encoder import ConceptParentRAGEncoder
    from ontoaligner.postprocess import rag_hybrid_postprocessor, rag_heuristic_postprocessor

    task = MaterialInformationMatOntoOMDataset()

    print("Test Task:", task)

    dataset = task.collect(
        source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="../assets/MI-MatOnto/matchings.xml"
    )

    encoder_model = ConceptParentRAGEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

    retriever_config = {
            "device":'cuda',
            "top_k": 5,
            "threshold": 0.1,
            # openai_key = "" # set your OpenAI key if you are willing to use open AI model as a retriever module of RAG.
    }
    llm_config={
        "device": "cuda",
        "max_length":300,
        "max_new_tokens":10,
        "huggingface_access_token": "", # if the interested LLM requires access via Huggingface
        "device_map": 'balanced',
        "batch_size": 15,
        "answer_set": {
                "yes": ["yes", "correct", "true", "positive", "valid"],
                "no": ["no", "incorrect", "false", "negative", "invalid"]
        }
        # "openai_key": "", # set your OpenAI key if you are willing to use open AI model as a LLM module of RAG.
    }

    model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)

    predicts = model.generate(input_data=encoded_ontology)

    # Heuristic postprocessor
    heuristic_matchings, heuristic_configs = rag_heuristic_postprocessor(predicts=predicts, topk_confidence_ratio=3, topk_confidence_score=3)
    evaluation = metrics.evaluation_report(predicts=heuristic_matchings, references=dataset['reference'])
    print("Heuristic Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Heuristic Matching Obtained Configuration:", heuristic_configs)

    # Hybrid postprocessor
    hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(predicts=predicts, ir_score_threshold=0.1, llm_confidence_th=0.8)
    evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
    print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Hybrid Matching Obtained Configuration:", hybrid_configs)


    # Convert matchings to XML format for compatibility with ontology alignment tools
    xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

    output_file_path = "matchings.xml"
    with open(output_file_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_str)

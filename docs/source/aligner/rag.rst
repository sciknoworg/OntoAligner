Retrieval Augmented Generation
================================

This tutorial walks you through the process of ontology matching using the OntoAligner library, leveraging retrieval-augmented generation (RAG) techniques. Starting with the necessary module imports, it defines a task and loads source and target ontologies along with reference matchings. The tutorial then encodes the ontologies using a specialized encoder, configures a retriever and an LLM, and generates predictions. Finally, it demonstrates two postprocessing techniques—heuristic and hybrid—followed by saving the matched alignments in XML format, ready for use or further analysis.

.. code-block:: python

    # Step1. Lets import Required Modules
    import json
    from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
    from ontoaligner.utils import metrics, xmlify
    from ontoaligner.aligner import MistralLLMBERTRetrieverRAG
    from ontoaligner.encoder import ConceptParentRAGEncoder
    from ontoaligner.postprocess import rag_hybrid_postprocessor, rag_heuristic_postprocessor

    #Step 2. Define the Task and Load Ontologies
    task = MaterialInformationMatOntoOMDataset()
    print("Test Task:", task)

    dataset = task.collect(
        source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="../assets/MI-MatOnto/matchings.xml"
    )

    # Step 3. Encode Ontologies
    encoder_model = ConceptParentRAGEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

    #Step 4. Configure the Retriever and LLM
    config = {
        "retriever_config": {"device": 'cuda', "top_k": 5, "threshold": 0.1},
        "llm_config": {
            "device": "cuda", "batch_size": 32,
            "answer_set": {"yes": ["yes", "true"], "no": ["no", "false"]}
        }
    }

    # Step 5. Generate Predictions
    model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
    model.load(llm_path = "mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")
    predicts = model.generate(input_data=encoded_ontology)

    # Step 6. Postprocess Matches
    # Heuristic Postprocessing
    heuristic_matchings, heuristic_configs = rag_heuristic_postprocessor(predicts=predicts, topk_confidence_ratio=3, topk_confidence_score=3)
    evaluation = metrics.evaluation_report(predicts=heuristic_matchings, references=dataset['reference'])
    print("Heuristic Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Heuristic Matching Obtained Configuration:", heuristic_configs)

    # Hybrid Postprocessing
    hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(predicts=predicts, ir_score_threshold=0.1, llm_confidence_th=0.8)
    evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
    print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))
    print("Hybrid Matching Obtained Configuration:", hybrid_configs)

    # Step 7. Save Matchings in XML Format
    xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

    output_file_path = "matchings.xml"
    with open(output_file_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_str)



In this tutorial, we demonstrated:

* Loading and encoding ontologies
* Using Mistral LLM with RAG for ontology matching
* Refining results with heuristic and hybrid postprocessing
* Saving results in XML format

You can customize the configurations and thresholds based on your specific dataset and use case. For more details, refer to the :doc:`../package_reference/postprocess`

FewShot RAG
------------------------
This tutorial works based on FewShot RAG matching, an extension of the RAG model, designed for few-shot learning tasks. The FewShot RAG workflow is the same as RAG but with two differences:

1. You only need to use FewShot encoders as follows, and since a fewshot model uses multiple examples you might also provide only specific examples from reference or other examples as a fewshot samples.

.. code-block:: python

    from ontoaligner.encoder import ConceptParentFewShotEncoder

    encoder_model = ConceptParentFewShotEncoder()
    encoded_ontology = encoder_model(source=dataset['source'],
                                     target=dataset['target'],
                                     reference=dataset['reference'])

2. Next, use a Fewshot Retrieval-Augmented Generation (RAG) model for ontology alignment.

.. code-block:: python

    from ontoaligner.aligner import MistralLLMBERTRetrieverFSRAG

    model = MistralLLMBERTRetrieverFSRAG(positive_ratio=0.7, n_shots=5, **config)

In-Context Vectors RAG
------------------------
This RAG variant performs ontology matching using ``ConceptRAGEncoder`` only. The In-Contect Vectors introduced by [1](https://github.com/shengliu66/ICV) tackle in-context learning as in-context vectors (ICV). We used LLMs in this perspective in the RAG module. The workflow is the same as RAG or FewShot RAG with the following differences:


1. Incorporate the ``ConceptRAGEncoder`` and also provide reference (or examples to build up the ICV vectors).

.. code-block:: python

    from ontoaligner.encoder import ConceptRAGEncoder
    encoder_model = ConceptRAGEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'], reference=dataset['reference'])

2. Next, import an ICVRAG model, here we use Falcon model:

.. code-block:: python

    from ontoaligner.aligner import FalconLLMBERTRetrieverICVRAG
    model = FalconLLMBERTRetrieverICVRAG(**config)

    model.load(llm_path="tiiuae/falcon-7b", ir_path="all-MiniLM-L6-v2")


[1] Liu, S., Ye, H., Xing, L., & Zou, J. (2023). `In-context vectors: Making in context learning more effective and controllable through latent space steering <https://arxiv.org/abs/2311.06668>`_. arXiv preprint arXiv:2311.06668.

Fewshot RAG
===============

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

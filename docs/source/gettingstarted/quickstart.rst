Quickstart
===========

User Guide
-------------

Characteristics of OntoAligner (a.k.a Ontology Alignment) Toolkit:

1. Ontology Matching Tasks Support: The ontology matching task, which aligns concepts between two ontologies. It uses a specific dataset (e.g ``MaterialInformationMatOntoOMDataset``) that contains source, target, and reference matching information.
2. Multi-Step Pipeline: The process is organized into clear steps, including task definition, data collection, encoding, matching and prediction, postprocessing, evaluation, and output saving. Each step uses modular components from the **OntoAligner** library..
3. Integration with Advanced Models: Applicable for a **wide range of models** with five distinct categories such as Lightweight, Retrieval, Large Language Models, Retrieval Augmented Generation, FewShots RAG, and In-Context Vectors RAG.
4. Postprocessing and Evaluation: Different postprocessing methods are implemented to refine the predicted matches.
5. Export Capability: The final matching results are saved in a standardized XML format or JSON or will be returned, ensuring compatibility with external tools or workflows that use tools like XML for ontology alignment.


Once you have `installed <installation.html>`_ OntoAligner, you can easily use OntoAligner models:

.. code-block:: python

    import ontoaligner

    task = ontoaligner.ontology.MaterialInformationMatOntoOMDataset()

    dataset = task.collect(source_ontology_path="source_ontology.xml",
                           target_ontology_path="target_ontology.xml",
                           reference_matching_path="reference.xml" )

    encoder_model = ontoaligner.encoder.ConceptRAGEncoder()
    encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

    retriever_config = {"top_k": 5}
    llm_config = {"max_new_tokens": 10, "batch_size": 32}

    model = ontoaligner.aligner.MistralLLMBERTRetrieverRAG(retriever_config=retriever_config,
                                                                     llm_config=llm_config)
    model.load(llm_path="mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")
    predicts = model.generate(input_data=encoded_ontology)

    matchings, _ = ontoaligner.postprocess.rag_hybrid_postprocessor(predicts=predicts,
                                                                    ir_score_threshold=0.5,
                                                                    llm_confidence_th=0.8)
    evaluation = ontoaligner.utils.metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
    print("Matching Evaluation Report:", evaluation)

    xml_str = ontoaligner.utils.xmlify.xml_alignment_generator(matchings=matchings)
    open("matchings.xml", "w", encoding="utf-8").write(xml_str)


.. sidebar:: Useful links:

    * `Package Reference > Ontology <../package_reference/parsers.html>`_
    * `Package Reference > Pipeline <../package_reference/pipeline.html>`_
    * `Package Reference > Encoder <../package_reference/encoders.html>`_
    * `Package Reference > Post-Process > Process <../package_reference/postprocess.html#module-ontoaligner.postprocess.process>`_
    * `Aligners > Lightweight Aligner <../aligner/lightweight.html>`_
    * `Aligners > Retrieval Aligner <../aligner/retriever.html>`_
    * `Aligners > Large Language Models Aligner <../aligner/llm.html>`_
    * `Aligners > Retrieval Augmented Generation <../aligner/rag.html>`_



With ``ontoaligner`` library, we perform ontology matching between a source and target ontology with loading a dataset using the `MaterialInformationMatOntoOMDataset <../package_reference/parsers.html#material-sciences-and-engineering-track>`_ class that considers ``MaterialInformation`` as source ontology, and ``MatOnto`` as target ontology. This followed by encoding the ontologies using `ConceptRAGEncoder <../package_reference/encoders.html#retrieval-augmented-generation-encoders>`_. Next, using ``MaterialInformationMatOntoOMDataset`` that configures a retriever and a large language model (LLM) for generating predictions, using a pre-trained Mistral-7B (``mistralai/Mistral-7B-v0.3``) model and an Sentence Transformer model (``all-MiniLM-L6-v2``). After RAG module prediction for matching, a ``hybrid`` postprocessing is applied to filter and refine the predictions , and evaluates the resulting matchings against a reference set. Finally, it generates an evaluation report and exports the matching results as an XML file. This process automates ontology alignment, making it easier to compare and merge knowledge structures. The postprocessing, a cardinality based filter runes using IR threshold to filter retriever outputs and then applies LLM based confidence score filtering.

Aligner Pipeline
--------------------------

.. sidebar:: Useful links:

    * `Package Reference > Pipeline <../package_reference/pipeline.html>`_


Characteristics of OntoAligner Pipeline:

1. It performs ontology alignment in one go.
2. Supports Lightweight matching, Retriever-based matching, LLM-based matching, and RAG (Retriever-Augmented Generation) variants.
3. It is a dataset first approach, it means you load your  alignment task then you choose the model that you like to do the matchings.

The usage for ``OntoAlignerPipeline``:


.. code-block:: python

   import ontoaligner

   pipeline = ontoaligner.OntoAlignerPipeline(
            task_class=ontoaligner.ontology.MaterialInformationMatOntoOMDataset,
            source_ontology_path="source_ontology.xml",
            target_ontology_path="target_ontology.xml",
            reference_matching_path="reference.xml"
            output_dir="results",
            output_format="xml"
        )

   matchings, evaluation = pipeline(method="rag",
                    llm_path='mistralai/Mistral-7B-v0.3',
                    retriever_path='all-MiniLM-L6-v2',
                    model_class=ontoaligner.aligner.MistralLLMBERTRetrieverRAG,
                    device='cuda',
                    batch_size=15,
                    return_matching=True,
                    evaluate=True,
                    save_matchings=False
                )
   # evaluation output:
   # {'intersection': 85,
   #  'precision': 47.22222222222222,
   #  'recall': 28.14569536423841,
   #  'f-score': 35.26970954356846,
   #  'predictions-len': 180,
   #  'reference-len': 302}



Next Steps
----------

Consider reading one of the following sections next:

* `Aligners > Lightweight Aligner <../aligner/lightweight.html>`_
* `Aligners > Retrieval Aligner <../aligner/retriever.html>`_
* `Aligners > Large Language Models Aligner <../aligner/llm.html>`_
* `Aligners > Retrieval Augmented Generation <../aligner/rag.html>`_
* `Aligners > FewShot RAG <../aligner/rag.html#fewshot-rag>`_
* `Aligners > In-Context Vectors RAG <../aligner/rag.html#in-context-vectors-rag>`_
* `Package Reference > Pipeline <../package_reference/pipeline.html>`_
* `Package Reference > Aligners <./package_reference/aligners.html>`_

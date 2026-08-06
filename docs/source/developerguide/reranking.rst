Reranking
=====================================================

.. sidebar:: Useful links:

    * `Retrieval Aligner > Reranking <https://ontoaligner.readthedocs.io/aligner/retriever.html#reranking>`_
    * `Developer Guide > Pipeline <pipeline.html>`_


This guide shows how reranking can be used as a reusable candidate-refinement step across OntoAligner workflows.
Reranking is not tied to one specific aligner. It can be applied after any component
that produces multiple target candidates for the same source concept.

The examples cover the main OntoAligner output styles, including grouped retrieval
candidates, graph-based candidates, RAG-style outputs, and flat predictions from OLaLA,
ensemble, FLORA, LLM-style, and custom workflows. In each case, reranking can be applied
when multiple target candidates are available for a source concept, either directly as
grouped candidates or after grouping flat predictions.

.. note::

    Reranking is useful only when multiple target candidates are available for a source.
    Single-target outputs, such as PropMatch or fuzzy lightweight results, are usually
    not suitable after final selection.


Usage
----------------------------

.. tab:: ⚙️ Setup

    The examples use the MaterialInformation-MatOnto dataset and a CrossEncoder
    reranker.

    .. code-block:: python

        # Import required modules
        import torch

        from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
        from ontoaligner.aligner import CrossEncoderReranking
        from ontoaligner.postprocess import retriever_postprocessor

        # Load source and target ontologies
        task = MaterialInformationMatOntoOMDataset()

        dataset = task.collect(
            source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
            target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
            reference_matching_path="assets/MI-MatOnto/matchings.xml",
        )

        # Select runtime device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize and load the reranker
        reranker = CrossEncoderReranking(
            device=device,
            top_k=5,
            normalize_score="sigmoid",
        )

        reranker.load(
            path="cross-encoder/ms-marco-MiniLM-L6-v2",
        )

    The helper below is used when an aligner returns flat
    ``source``-``target``-``score`` predictions.

    .. code-block:: python

        # Convert flat predictions into grouped candidate format
        def group_predictions_for_reranking(predictions):
            grouped_predictions = {}

            for prediction in predictions:
                source = prediction["source"]
                target = prediction["target"]
                score = prediction.get("score", 0.0)

                if source not in grouped_predictions:
                    grouped_predictions[source] = {
                        "source": source,
                        "target-cands": [],
                        "score-cands": [],
                    }

                grouped_predictions[source]["target-cands"].append(target)
                grouped_predictions[source]["score-cands"].append(float(score))

            return list(grouped_predictions.values())


.. tab:: 🔁 Grouped Output Reranking

    Use this pattern when the aligner already returns grouped candidates with
    ``target-cands`` and ``score-cands``.

    .. code-block:: python

        # Encode source and target concepts
        from ontoaligner.encoder import ConceptParentLightweightEncoder
        from ontoaligner.aligner import SBERTRetrieval

        encoder_model = ConceptParentLightweightEncoder()

        source_onto, target_onto = encoder_model(
            source=dataset["source"],
            target=dataset["target"],
        )

        # Generate grouped retrieval candidates
        retriever = SBERTRetrieval(
            device=device,
            top_k=10,
        )

        retriever.load(
            path="all-MiniLM-L6-v2",
        )

        retrieval_outputs = retriever.generate(
            input_data=[
                source_onto,
                target_onto,
            ]
        )

        # Rerank the grouped candidates
        reranked_outputs = reranker.generate(
            input_data=[
                source_onto,
                target_onto,
                retrieval_outputs,
            ]
        )

        # Convert reranked candidates into final matchings
        matchings = retriever_postprocessor(
            predicts=reranked_outputs,
            threshold=0.5,
        )

    .. note::

        This pattern applies to retrieval-style aligners and other outputs that already
        use ``target-cands`` and ``score-cands``.


.. tab:: 🧩 Flat Output Reranking

    Use this pattern when a workflow produces final flat ``source``-``target``-``score``
    predictions. For RAG-style workflows, apply the RAG postprocessor first and then
    group the final matchings for reranking.

    .. code-block:: python

        # Apply RAG postprocessing to produce final flat matchings
        flat_predictions, configs = rag_hybrid_postprocessor(
            predicts=predicts,
            ir_score_threshold=0.5,
            llm_confidence_th=0.8,
        )

        # Group flat matchings for reranking
        grouped_candidates = group_predictions_for_reranking(
            predictions=flat_predictions,
        )

        # Rerank grouped candidates
        reranked_outputs = reranker.generate(
            input_data=[
                source_onto,
                target_onto,
                grouped_candidates,
            ]
        )

        # Convert reranked candidates into final matchings
        matchings = retriever_postprocessor(
            predicts=reranked_outputs,
            threshold=0.5,
        )

    For direct flat outputs, such as OLaLA, the output from ``generate()`` can be grouped
    directly before reranking.

    .. code-block:: python

        # Generate flat OLaLA alignments
        flat_predictions = olala.generate(
            input_data=encoded_ontology,
        )

        # Group flat alignments for reranking
        grouped_candidates = group_predictions_for_reranking(
            predictions=flat_predictions,
        )

    .. note::

        This pattern applies to OLaLA, ensemble outputs, FLORA-style outputs, and final
        flat outputs from RAG, FewShotRAG, ICV, standalone LLM workflows, or custom
        aligners.

.. tab:: 🕸️ Graph Candidate Reranking

    Graph candidate reranking applies when a graph-based aligner keeps multiple target
    candidates per source. In the tutorial, ``ConvEAligner`` is used with
    ``retriever=True``.

    .. code-block:: python

        # Load graph ontology data
        from ontoaligner.ontology import GraphTripleOMDataset
        from ontoaligner.encoder import GraphTripleEncoder, ConceptParentLightweightEncoder
        from ontoaligner.aligner import ConvEAligner

        graph_task = GraphTripleOMDataset(
            ontology_name="MI-MatOnto",
        )

        graph_dataset = graph_task.collect(
            source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
            target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
            reference_matching_path="assets/MI-MatOnto/matchings.xml",
        )

        # Encode graph triples
        graph_encoder = GraphTripleEncoder()

        encoded_graph_dataset = graph_encoder(**graph_dataset)

        # Encode source and target text for reranking
        text_task = MaterialInformationMatOntoOMDataset()

        text_dataset = text_task.collect(
            source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
            target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
            reference_matching_path="assets/MI-MatOnto/matchings.xml",
        )

        text_encoder = ConceptParentLightweightEncoder()

        source_onto, target_onto = text_encoder(
            source=text_dataset["source"],
            target=text_dataset["target"],
        )

        # Generate graph-based candidate outputs
        aligner = ConvEAligner(
            model="ConvE",
            device="cpu",
            embedding_dim=300,
            num_epochs=3,
            train_batch_size=128,
            eval_batch_size=64,
            num_negs_per_pos=5,
            random_seed=42,
            retriever=True,
            top_k=10,
        )

        graph_candidates = aligner.generate(
            input_data=encoded_graph_dataset,
        )

        # Rerank graph-generated candidates
        reranked_graph_candidates = reranker.generate(
            input_data=[
                source_onto,
                target_onto,
                graph_candidates,
            ]
        )

        # Convert reranked candidates into final matchings
        reranked_graph_matchings = retriever_postprocessor(
            predicts=reranked_graph_candidates,
            threshold=0.3,
        )

    .. note::

        This pattern applies to graph-based aligners when candidate retrieval is enabled.
        When ``retriever=False``, the output may not contain multiple candidates to rerank.


.. tab:: 🧠 RAG IR-Output Reranking

    RAG-style aligners produce IR candidates before LLM verification. These
    ``ir-outputs`` are grouped candidates, so they can be reranked directly before being
    sent to the LLM.

    .. code-block:: python

        # Encode source and target concepts for RAG
        from ontoaligner.encoder import ConceptParentRAGEncoder
        from ontoaligner.aligner.rag.rag import RAG, AutoModelDecoderRAGLLMV2
        from ontoaligner.aligner.retrieval.models import SBERTRetrieval
        from ontoaligner.postprocess import rag_hybrid_postprocessor

        encoder_model = ConceptParentRAGEncoder()

        encoded_ontology = encoder_model(
            source=dataset["source"],
            target=dataset["target"],
        )

        # Initialize and load the RAG aligner
        model = RAG(
            retriever=SBERTRetrieval,
            llm=AutoModelDecoderRAGLLMV2,
            retriever_config={
                "device": device,
                "top_k": 10,
                "threshold": 0.1,
            },
            llm_config={
                "device": "cpu",
                "max_length": 300,
                "max_new_tokens": 10,
                "batch_size": 15,
                "answer_set": {
                    "yes": ["yes", "correct", "true", "positive", "valid"],
                    "no": ["no", "incorrect", "false", "negative", "invalid"],
                },
            },
        )

        model.load(
            llm_path="distilgpt2",
            ir_path="all-MiniLM-L6-v2",
        )

        # Generate IR candidate outputs
        retrieval_input = encoded_ontology["retriever-encoder"]()(
            **encoded_ontology["task-args"]
        )

        source_onto = retrieval_input[0]
        target_onto = retrieval_input[1]

        ir_outputs = model.Retrieval.generate(
            input_data=retrieval_input,
        )

        # Rerank IR candidates before LLM verification
        reranked_ir_outputs = reranker.generate(
            input_data=[
                source_onto,
                target_onto,
                ir_outputs,
            ]
        )

        # Convert reranked IR candidates into source-target pairs
        reranked_ir_matchings = retriever_postprocessor(
            predicts=reranked_ir_outputs,
            threshold=0.5,
        )

        # Verify reranked candidates with the LLM
        llm_predictions = model.llm_generate(
            input_data=encoded_ontology,
            ir_output=reranked_ir_matchings,
        )

        # Apply RAG postprocessing
        predicts = [
            {"ir-outputs": reranked_ir_outputs},
            {"llm-output": llm_predictions},
        ]

        matchings, configs = rag_hybrid_postprocessor(
            predicts=predicts,
            ir_score_threshold=0.5,
            llm_confidence_th=0.8,
        )

    .. note::

        This pattern applies to RAG, FewShotRAG, ICV, and custom RAG-style workflows
        when retrieval candidates are reranked before LLM verification.

Configuration
----------------------------

Reranking can be applied through different workflow patterns depending on the aligner
output format:

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Aligner output
     - Before reranking
     - After reranking
   * - Grouped output
     - Use ``target-cands`` and ``score-cands`` directly.
     - Apply ``retriever_postprocessor``.
   * - Flat output
     - Group flat ``source``-``target``-``score`` predictions by source.
     - Apply ``retriever_postprocessor``.
   * - Graph candidate output
     - Use ``retriever=True`` and encode source and target text for reranking.
     - Apply ``retriever_postprocessor``.
   * - RAG IR output
     - Use ``ir-outputs`` directly before LLM verification.
     - Run LLM verification and RAG postprocessing.
   * - RAG / FewShotRAG / ICV / LLM final output
     - Apply the model-specific postprocessor, then group flat predictions by source.
     - Apply ``retriever_postprocessor``.
   * - ``AlignerPipeline``
     - Configure ``reranker`` and ``postprocessor`` during pipeline initialization.
     - The configured postprocessor is applied internally.

.. note::

    A complete tutorial notebook is available at
    `tutorial/05-reusable-reranking-in-ontoaligner.ipynb <https://github.com/sciknoworg/OntoAligner/blob/dev/tutorial/05-reusable-reranking-in-ontoaligner.ipynb>`_.
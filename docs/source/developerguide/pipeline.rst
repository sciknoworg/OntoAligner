Pipeline
=====================================================

.. sidebar:: Useful links:

    * `Aligners > Ensemble Learning <../aligner/ensemble_learning.html>`_
    * `Aligners > Lightweight Aligner <../aligner/lightweight.html>`_
    * `Aligners > Retrieval Aligner <../aligner/retriever.html>`_
    * `Aligners > Large Language Models Aligner <../aligner/llm.html>`_
    * `Aligners > Retrieval Augmented Generation <../aligner/rag.html>`_
    * `Aligners > Knowledge Graph Embedding Aligner <../aligner/kge.html>`_


``AlignerPipeline`` provides a reusable execution flow for running one user-provided
encoder and one ontology matching aligner over a collected ontology matching dataset.
It is useful when users want direct control over the encoder, aligner, model loading,
LLM dataset batching, and optional postprocessing.

Unlike a full orchestration pipeline, :class:`AlignerPipeline` does not collect
datasets, choose methods, define model-specific configurations, evaluate predictions,
or save outputs. It focuses only on running the configured encoder-aligner setup and
returning predictions.

Given two ontologies :math:`O_1` and :math:`O_2`, :class:`AlignerPipeline` produces
a list of correspondence predictions through four stages:

**🔧 1. Component Setup**: Provide the encoder, aligner, dataset, and optional
pipeline settings such as ``load_params``, ``llm_dataset_class``, ``postprocessor``,
or ``postprocessor_params``.

**⚙️ 2. Encoding**: Convert the collected ontology matching dataset into the format
expected by the aligner.

**🧠 3. Prediction Generation**: Generate predictions from encoded ontology data, with
optional LLM dataset batching when ``llm_dataset_class`` is provided.

**🧹 4. Optional Postprocessing**: Apply a user-provided postprocessor to convert,
filter, or normalize predictions before returning the final pipeline output.

Usage
----------

This module guides you through a step-by-step process for running a single ontology alignment model using :class:`AlignerPipeline`. By the end, you’ll understand how to collect an ontology matching dataset, configure an encoder and aligner, generate predictions, evaluate results, and save the outputs in XML and JSON formats.

.. tab:: ➡️ 1: Import

    Import the dataset class, encoder, aligner, pipeline, and utility modules.

    .. code-block:: python

        import json

        from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
        from ontoaligner.utils import metrics, xmlify
        from ontoaligner.encoder import ConceptParentLightweightEncoder
        from ontoaligner.aligner import SimpleFuzzySMLightweight
        from ontoaligner import AlignerPipeline

.. tab:: ➡️ 2: Parse Ontologies

    Load the source ontology, target ontology, and reference alignment using an OntoAligner dataset class.

    .. code-block:: python

        task = MaterialInformationMatOntoOMDataset()
        print("Test Task:", task)

        dataset = task.collect(
            source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
            target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
            reference_matching_path="assets/MI-MatOnto/matchings.xml",
        )

    The collected dataset contains the source ontology items, target ontology items, and optional reference matchings.

    .. code-block::

        {
            "source": [...],
            "target": [...],
            "reference": [...]
        }

.. tab:: ➡️ 3: Configure AlignerPipeline

    Configure :class:`AlignerPipeline` with one encoder, one aligner, and the collected ontology matching dataset.

    .. code-block:: python

        aligner_pipeline = AlignerPipeline(
            encoder=ConceptParentLightweightEncoder(),
            aligner=SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2),
            om_dataset=dataset,
        )

    The encoder prepares the ontology items for the aligner. The aligner then generates candidate correspondences from the encoded data.

.. tab:: ➡️ 4: Generate

    Call ``generate()`` to encode the dataset and generate predictions.

    .. code-block:: python

        matchings = aligner_pipeline.generate()

    The output is a list of flat source-target correspondences.

    .. code-block::

        [
            {"source": "...", "target": "...", "score": 0.9},
            ...
        ]

    ``generate()`` can also receive a dataset directly through ``input_data``. If ``input_data`` is provided, it is used instead of the dataset stored in ``om_dataset``.

    .. code-block:: python

        matchings = aligner_pipeline.generate(input_data=dataset)

.. tab:: ➡️ 5: Evaluate and Export

    Compare predictions to a reference alignment and export results.

    .. code-block:: python

        # Evaluate
        evaluation = metrics.evaluation_report(
            predicts=matchings,
            references=dataset["reference"],
        )

        print("Aligner Pipeline Evaluation Report:")
        print(json.dumps(evaluation, indent=4))

    Example output:

    .. code-block::

        {
            "intersection": 42,
            "precision": 7.706422018348624,
            "recall": 13.90728476821192,
            "f-score": 9.917355371900827,
            "predictions-len": 545,
            "reference-len": 302
        }

    Export the final alignment to XML (OAEI-compatible) or JSON:

    .. tab:: 📄 Export to XML

        .. code-block:: python

            xml_str = xmlify.xml_alignment_generator(matchings=matchings)
            with open("aligner_pipeline_matchings.xml", "w", encoding="utf-8") as f:
                f.write(xml_str)

    .. tab:: 🧾 Export to JSON

        .. code-block:: python

            with open("aligner_pipeline_matchings.json", "w", encoding="utf-8") as f:
                json.dump(matchings, f, indent=4, ensure_ascii=False)

.. note::

        A complete aligner pipeline example is available at
        `examples/aligner_pipeline.py <https://github.com/sciknoworg/OntoAligner/blob/dev/examples/aligner_pipeline.py>`_.

Configuration
--------------------

.. tab:: 🔧 AlignerPipeline

    .. list-table::
       :header-rows: 1
       :widths: 22 12 14 52

       * - Parameter
         - Type
         - Default
         - Description
       * - **encoder**
         - BaseEncoder
         - —
         - Encoder model used to encode the ontology matching dataset.
       * - **aligner**
         - BaseOMModel
         - —
         - Ontology matching aligner used to generate predictions.
       * - **om_dataset**
         - dict
         - ``None``
         - Pre-collected ontology matching dataset.
       * - **load_params**
         - dict
         - ``None``
         - Parameters forwarded to the aligner ``load`` method.
       * - **llm_dataset_class**
         - Dataset
         - ``None``
         - Dataset class used to wrap LLM inputs.
       * - **batch_size**
         - int
         - ``1``
         - Batch size used for LLM dataset generation.
       * - **shuffle**
         - bool
         - ``False``
         - Whether to shuffle LLM dataset batches.
       * - **postprocessor**
         - Any
         - ``None``
         - Optional postprocessor applied to pipeline predictions.
       * - **postprocessor_params**
         - dict
         - ``None``
         - Parameters forwarded to the postprocessor.
       * - **include_reference**
         - bool
         - ``False``
         - Whether to pass reference matchings to the encoder.
       * - ****kwargs**
         - dict
         - ``{}``
         - Additional keyword arguments forwarded to the base ontology matching model.

Configuration Example:

.. code-block:: python

    #FewShotRAG
    AlignerPipeline(
            encoder=ConceptParentFewShotEncoder(),
            aligner=MistralLLMBERTRetrieverFSRAG(
                positive_ratio=1.0,
                n_shots=1,
                retriever_config=retriever_config,
                llm_config=llm_config,
            ),
            om_dataset=dataset,
            load_params={
                "llm_path": llm_model_path,
                "ir_path": ir_model_path,
            },
            postprocessor=rag_heuristic_postprocessor,
            postprocessor_params={
                "topk_confidence_ratio": 3,
                "topk_confidence_score": 3,
            },
            include_reference=True,
        )

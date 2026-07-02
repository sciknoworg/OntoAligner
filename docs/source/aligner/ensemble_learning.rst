Ensemble Learning Aligner
=====================================================

Ensemble Learning
------------------
.. sidebar:: Useful links:

    * `Developer Guide > AlignerPipeline <../developerguide/pipeline.html>`_

**Ensemble Learning** combines predictions from multiple ontology alignment pipelines to produce a final set of correspondences. In OntoAligner, ensemble learning is handled by :class:`EnsembleLearningAligner`, where each ensemble member is represented as a branch configured with :class:`AlignerPipeline`.

Each branch follows the standard OntoAligner flow: encode the ontology matching dataset, load the aligner when needed, generate predictions, and optionally apply branch-level postprocessing. Postprocessing may be required before voting for LLM, RAG, and KGE outputs because these aligners may produce outputs that need conversion or filtering before fusion.

.. hint::

    **Why Ensemble Learning for Ontology Alignment?**

    1) *Complementary Signals*: Combines lexical similarity, semantic retrieval, graph structure, and LLM/RAG-based verification from different aligners.
    2) *Robustness*: Reduces reliance on a single aligner, which can help balance the weaknesses of individual models.
    3) *Model-Agnostic Fusion*: Allows heterogeneous aligner families to contribute through a shared voting strategy and produce one final alignment.

.. raw:: html

    <div align="center">
        <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/refs/heads/dev/docs/source/img/ensemble.PNG" width="70%"/>
    </div>

The ensemble workflow has four stages:

**🔧 1. Branch Configuration**: Multiple :class:`AlignerPipeline` are configured with encoders, aligners, datasets, optional loading parameters, and optional postprocessors.

**⚙️ 2. Branch Prediction**: Each branch generates correspondences independently using lightweight, retrieval, KGE, LLM, or RAG-based aligners.

**🧩 3. Output Normalization**: Branch outputs are converted into a common ``source``-``target``-``score`` format before fusion.

**🗳️ 4. Voting**: A voting method combines weighted branch outputs into the final matchings.

Usage
---------
This module guides you through a step-by-step process for performing ensemble-based ontology alignment using multiple OntoAligner models. By the end, you’ll understand how to configure aligner pipeline, combine their predictions with voting strategies, evaluate the final matchings, and save the outputs in XML and JSON formats.

.. tab:: ➡️ 1: Import

    Import the dataset classes, encoders, aligners, postprocessors, ensemble aligner,
    and voting strategy.

    .. code-block:: python

        import json
        import torch

        from sklearn.linear_model import LogisticRegression

        from ontoaligner.ontology import MaterialInformationMatOntoOMDataset, GraphTripleOMDataset
        from ontoaligner.utils import metrics, xmlify
        from ontoaligner.encoder import (
            ConceptParentLightweightEncoder,
            ConceptLLMEncoder,
            ConceptParentRAGEncoder,
            GraphTripleEncoder,
        )
        from ontoaligner.aligner import (
            SimpleFuzzySMLightweight,
            SBERTRetrieval,
            AutoModelDecoderLLM,
            ConceptLLMDataset,
            MistralLLMBERTRetrieverRAG,
            TransEAligner,
        )
        from ontoaligner.postprocess import (
            TFIDFLabelMapper,
            llm_postprocessor,
            graph_postprocessor,
            rag_heuristic_postprocessor,
        )
        from ontoaligner.aligner.ensemble import EnsembleLearningAligner
        from ontoaligner.aligner.ensemble.voting import ReciprocalRankFusionVoting
        from ontoaligner import AlignerPipeline

.. tab:: ➡️ 2: Parse Ontologies

    Load the source ontology, target ontology, and reference alignment using OntoAligner
    dataset classes.

    .. code-block:: python

        source_ontology_path = "assets/MI-MatOnto/mi_ontology.xml"
        target_ontology_path = "assets/MI-MatOnto/matonto_ontology.xml"
        reference_matching_path = "assets/MI-MatOnto/matchings.xml"

        task = MaterialInformationMatOntoOMDataset()
        print("Test Task:", task)

        dataset = task.collect(
            source_ontology_path=source_ontology_path,
            target_ontology_path=target_ontology_path,
            reference_matching_path=reference_matching_path,
        )

        graph_dataset = GraphTripleOMDataset().collect(
            source_ontology_path,
            target_ontology_path,
            reference_matching_path,
        )

.. tab:: ➡️ 3: Configure Ensemble

    Configure the runtime settings, model paths, label mapper, RAG configuration,
    and ensemble aligners. Each branch is represented by an :class:`AlignerPipeline`
    and may include branch-level postprocessing before voting.

    .. code-block:: python

        device = "cuda" if torch.cuda.is_available() else "cpu"

        ir_model_path = "all-MiniLM-L6-v2"
        llm_model_path = "Qwen/Qwen2.5-1.5B-Instruct"

        mapper = TFIDFLabelMapper(
            classifier=LogisticRegression(),
            ngram_range=(1, 1),
            label_dict={
                "yes": ["yes", "correct", "true", "same", "equivalent", "valid"],
                "no": ["no", "incorrect", "false", "different", "not same", "invalid"],
            },
        )

        retriever_config = {
            "device": device,
            "top_k": 5,
            "threshold": 0.1,
        }

        llm_config = {
            "device": device,
            "max_length": 300,
            "max_new_tokens": 10,
            "batch_size": 1,
            "answer_set": {
                "yes": ["yes", "correct", "true", "positive", "valid"],
                "no": ["no", "incorrect", "false", "negative", "invalid"],
            },
        }

        aligners = [
            (
                "lightweight",
                AlignerPipeline(
                    encoder=ConceptParentLightweightEncoder(),
                    aligner=SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2),
                    om_dataset=dataset,
                ),
                1.0,
            ),
            (
                "sbert",
                AlignerPipeline(
                    encoder=ConceptParentLightweightEncoder(),
                    aligner=SBERTRetrieval(device=device, top_k=5),
                    om_dataset=dataset,
                    load_params={"path": ir_model_path},
                ),
                1.0,
            ),
            (
                "kge",
                AlignerPipeline(
                    encoder=GraphTripleEncoder(),
                    aligner=TransEAligner(
                        model="TransE",
                        device=device,
                        embedding_dim=32,
                        num_epochs=1,
                        train_batch_size=32,
                        eval_batch_size=32,
                        num_negs_per_pos=1,
                        random_seed=42,
                    ),
                    om_dataset=graph_dataset,
                    postprocessor=graph_postprocessor,
                    postprocessor_params={"threshold": 0.0},
                ),
                1.0,
            ),
            (
                "llm",
                AlignerPipeline(
                    encoder=ConceptLLMEncoder(),
                    aligner=AutoModelDecoderLLM(
                        device=device,
                        max_length=300,
                        max_new_tokens=20,
                        batch_size=1,
                    ),
                    om_dataset=dataset,
                    llm_dataset_class=ConceptLLMDataset,
                    load_params={"path": llm_model_path},
                    postprocessor=llm_postprocessor,
                    postprocessor_params={
                        "mapper": mapper,
                        "interested_class": "yes",
                    },
                ),
                1.0,
            ),
            (
                "rag",
                AlignerPipeline(
                    encoder=ConceptParentRAGEncoder(),
                    aligner=MistralLLMBERTRetrieverRAG(
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
                ),
                1.0,
            ),

        ]

    Each branch is represented as a tuple containing the branch name, an
    :class:`AlignerPipeline`, and an optional branch weight.

    .. code-block:: python

        aligners = [
            ("lightweight", AlignerPipeline(...), 1.0),
            ("sbert", AlignerPipeline(...), 1.0),
        ]

    The branch weight controls how much influence the branch has during voting.

.. tab:: ➡️ 4: Ensemble Learning Aligner

    Initialize :class:`EnsembleLearningAligner` with the configured aligners and a voting
    method. The default voting method is :class:`ReciprocalRankFusionVoting`.

    .. code-block:: python

        ensemble = EnsembleLearningAligner(
            aligners=aligners,
            voting=ReciprocalRankFusionVoting(k=60),
        )

        final_matchings = ensemble.generate()

    The output is a list of flat source-target correspondences sorted by score.

    .. code-block::

        [
            {"source": "...", "target": "...", "score": 0.9},
            ...
        ]

.. tab:: ➡️ 5: Evaluate and Export

    Compare predictions to a reference alignment and export results.

    .. code-block:: python

        # Evaluate
        evaluation = metrics.evaluation_report(
            predicts=final_matchings,
            references=dataset["reference"],
        )
        print("Ensemble Learning Evaluation Report:")
        print(json.dumps(evaluation, indent=4))

    Example output:

    .. code-block::

        {
            "intersection": 154,
            "precision": 2.651058702014116,
            "recall": 50.993377483443716,
            "f-score": 5.040091638029782,
            "predictions-len": 5809,
            "reference-len": 302
        }

    Export the final alignment to XML (OAEI-compatible) or JSON:

    .. tab:: 📄 Export to XML

        .. code-block:: python

            xml_str = xmlify.xml_alignment_generator(matchings=final_matchings)
            with open("ensemble_matchings.xml", "w", encoding="utf-8") as f:
                f.write(xml_str)

    .. tab:: 🧾 Export to JSON

        .. code-block:: python

            with open("ensemble_matchings.json", "w", encoding="utf-8") as f:
                json.dump(final_matchings, f, indent=4, ensure_ascii=False)

.. note::

        A complete ensemble learning example is available at
        `examples/ensemble.py <https://github.com/sciknoworg/OntoAligner/blob/dev/examples/ensemble.py>`_.

Voting Strategies
-----------------------

Voting strategies combine normalized predictions from multiple aligners. Each branch
contributes a list of predictions and a branch weight. The branch weight controls the
influence of the branch during fusion.


.. list-table::
   :header-rows: 1
   :widths: 25 50 15

   * - Strategy
     - Description
     - Link
   * - ``ReciprocalRankFusionVoting``
     - Adds reciprocal-rank scores from each branch and ranks pairs by the fused score.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/aligner/ensemble/voting/reciprocal_rank_fusion.py>`_
   * - ``BordaCountVoting``
     - Assigns normalized rank-based points to predictions and sums them across aligners.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/aligner/ensemble/voting/borda.py>`_
   * - ``CondorcetVoting``
     - Compares target candidates pairwise for each source and scores candidates by pairwise wins.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/aligner/ensemble/voting/condorce.py>`_
   * - ``ScoreAverageVoting``
     - Computes the weighted average score for each source-target pair across aligners.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/aligner/ensemble/voting/average.py>`_
   * - ``WeightedVoting``
     - Counts weighted branch support for each source-target pair and filters by vote settings.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/aligner/ensemble/voting/weighted.py>`_

.. hint::

        Rank-based voting is useful for heterogeneous aligners where scores are not directly
        comparable. Score-based voting is useful when scores come from similar model families
        or are already comparable.

To use voting strategies:

Import a voting method and pass it to :class:`EnsembleLearningAligner`.

.. code-block:: python

    from ontoaligner.aligner.ensemble.voting import ReciprocalRankFusionVoting

    ensemble = EnsembleLearningAligner(
        aligners=aligners,
        voting=ReciprocalRankFusionVoting(k=60),
    )

A different voting method can be used by changing the import & voting object.

.. code-block:: python

    from ontoaligner.aligner.ensemble.voting import ScoreAverageVoting

    ensemble = EnsembleLearningAligner(
        aligners=aligners,
        voting=ScoreAverageVoting(),
    )



Configuration
--------------------

.. tab:: 🧩 EnsembleLearningAligner

    .. list-table::
       :header-rows: 1
       :widths: 14 12 14 60

       * - Parameter
         - Type
         - Default
         - Description
       * - **aligners**
         - list
         - —
         - A list of branch tuples in the form ``(name, aligner_pipeline)`` or
           ``(name, aligner_pipeline, weight)``. At least two aligner pipelines
           are required.
       * - **voting**
         - BaseVoting
         - ``ReciprocalRankFusionVoting()``
         - Voting method used to combine branch predictions.
       * - ****kwargs**
         - dict
         - ``{}``
         - Additional keyword arguments forwarded to the base ontology matching model.


.. tab:: 🗳️ ReciprocalRankFusionVoting

    .. list-table::
       :header-rows: 1
       :widths: 22 12 14 52

       * - Parameter
         - Type
         - Default
         - Description
       * - **k**
         - int
         - ``60``
         - Smoothing constant used in reciprocal rank fusion.

.. tab:: ✅ WeightedVoting

    .. list-table::
       :header-rows: 1
       :widths: 22 12 14 52

       * - Parameter
         - Type
         - Default
         - Description
       * - **min_votes**
         - int
         - ``1``
         - Minimum number of aligners required for a pair.
       * - **score_threshold**
         - float
         - ``None``
         - Minimum branch score required to count a vote.

    ``WeightedVoting`` can work as majority voting when all aligners have the same
    weight and ``min_votes`` is set to more than half of the total number of aligners;
    ``score_threshold`` is optional.

    Example use when the count of aligners is 5:

    .. code-block:: python

        ensemble = EnsembleLearningAligner(
            aligners=aligners,
            voting=WeightedVoting(min_votes=3),
        )

.. note::

    For details on configuring :class:`AlignerPipeline` & :class:`EnsembleLearningAligner`, see:

    * `Developer Guide > AlignerPipeline Configuration <../developerguide/pipeline.html#configuration>`_
    * `Package Reference > Ensemble Aligner <../package_reference/aligners.html#ensemble-aligner>`_

No additional constructor parameters are required for BordaCountVoting, CondorcetVoting, ScoreAverageVoting.

Configuration Example:

.. code-block:: python

    ensemble = EnsembleLearningAligner(
        aligners=aligners,
        voting=ReciprocalRankFusionVoting(k=60),
    )

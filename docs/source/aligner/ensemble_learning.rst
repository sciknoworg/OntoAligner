Ensemble Learning Aligner
=====================================================

Ensemble Learning
------------------

.. sidebar:: Useful Links

    * `Developer Guide > AlignerPipeline <../developerguide/pipeline.html>`_
    * `API Reference > EnsembleLearningAligner <../api/ensemble.html>`_

**Ensemble Learning** combines predictions from multiple heterogeneous ontology alignment pipelines into a unified set of correspondences. Managed by :class:`EnsembleLearningAligner`, each constituent member is wrapped inside an :class:`AlignerPipeline`. Because the framework is **aligner-agnostic**, any model capable of outputting source--target correspondences can participate—including lexical matchers, structural/graph-based algorithms, semantic retrieval systems, LLMs, RAG, etc.

.. hint::

    **Why Use Ensemble Learning for Ontology Alignment?**

    1. **Complementary Signals:** Combines distinct alignment heuristics (lexical surface form, structural proximity, deep vector embeddings, and LLM semantic reasoning).
    2. **Increased Robustness:** Mitigates individual model weaknesses, reducing false positives and out-of-vocabulary blind spots.
    3. **Model-Agnostic Fusion:** Integrates heterogeneous matchers into a single workflow via unified score/rank fusion and post-fusion decision policies.

.. raw:: html

    <div align="center">
        <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/refs/heads/dev/docs/source/img/ensemble_learning_aligner.png" width="75%" alt="OntoAligner Ensemble Architecture Overview"/>
    </div>

Overall, the ensemble execution operates in a structured **two-stage decision process** across four sequential steps:

.. tab:: 1. 🔧 Ensemble Representation & Input Setup

	An ensemble consists of :math:`n` constituent pipelines  :math:`A = \{A_1, A_2, \dots, A_n\}`, where each aligner pipeline :math:`A_i` is assigned a reliability weight :math:`w_i \in \mathbb{R}` (default :math:`w_i = 1`).

	Each aligner independently generates candidate correspondences over source (:math:`O_{\text{source}}`) and target (:math:`O_{\text{target}}`) ontologies:

	.. math::

	   P_i = \{ (s, t, q_i(s, t)) \mid s \in O_{\text{source}}, \, t \in O_{\text{target}} \}

	where :math:`q_i(s,t) \in [0, 1]` represents the confidence score assigned to pair :math:`(s,t)` by aligner :math:`A_i`. The total ensemble input state is defined as :math:`\mathcal{P} = \{ (P_i, w_i) \}_{i=1}^n`.


.. tab:: 2. 🧩 Output Normalization

	Outputs generated across heterogeneous aligners are normalized into flat source--target triples. For example, a retrieval-based mapping :math:`s \mapsto \{(t_1, q_1), (t_2, q_2), \dots\}` is expanded to individual triples :math:`(s, t_j, q_j)`.

	To guarantee a unique representation before voting, duplicate candidate pairs are resolved by retaining the entry with the highest confidence score:

	.. math::

	   q(s, t) = \arg\max q_i(s, t)

.. tab:: 🗳️ Stage 1: Voting-Based Fusion

	The fusion component aggregates predictions across all constituent pipelines. First, it constructs the candidate space :math:`\mathcal{C}` from the union of all candidate pairs proposed by at least one aligner:

	.. math::

	   \mathcal{C} = \bigcup_{i=1}^{n} \{(s, t) \mid (s, t, q_i(s,t)) \in P_i\}

	For each candidate pair :math:`(s,t) \in \mathcal{C}`, a voting strategy :math:`\mathcal{V}` computes a unified fused score :math:`S(s,t)` parameterized by strategy-specific parameters :math:`\theta`:

	.. math::

	   S(s,t) = \mathcal{V}\left( \{(P_i, w_i)\}_{i=1}^{n}, \, (s,t), \, \theta \right)

	The candidates are then sorted in descending order of their fused score to yield the ranked candidate pool $\mathcal{F}$:

	.. math::

	   \mathcal{F} = \operatorname{sort}_{\downarrow S}\left( \{(s,t,S(s,t)) \mid (s,t) \in \mathcal{C}\} \right)

	**Supported Voting Strategies** (:math:`\mathcal{V}`):

	* **Weighted Voting:** Aggregates weighted linear confidence scores.
	* **Reciprocal Rank Fusion (RRF):** Fuses candidates based on candidate position ranks rather than raw confidence scores.
	* **Borda Count:** Applies rank-position point totals across aligner preference lists.
	* **Condorcet Voting:** Evaluates pairwise preferences between candidate matchings.
	* **Score Averaging:** Unweighted mean confidence score across participating pipelines.

.. tab:: 4. 🎯 Stage 2: Post-Fusion Selection

	After fusion, a selection policy :math:`g_\phi` converts the ranked candidate pool :math:`\mathcal{F}` into the final target alignment :math:`\mathcal{A} = \operatorname{Select}(\mathcal{F}, g_\phi)`:

	* **Top-1 per Source:** Assigns each source entity to its highest-scoring target candidate: :math:`t^* = \arg\max_t S(s, t)`

	* **Top-k per Source (with optional Margin):** Retains the top-k target candidates per source entity. When a relative score margin :math:`m \in [0, 1]` is supplied, candidates must additionally satisfy: :math:`S(s,t) \ge m \cdot \max_{t'} S(s, t')`

	* **Threshold-based Selection:** Retains only correspondences exceeding an absolute confidence threshold :math:`\gamma`: :math:`\mathcal{A} = \{ (s, t) \in \mathcal{F} \mid S(s,t) \ge \gamma \}`

	* **Greedy Bijective Selection:** Enforces strict one-to-one (1:1) mapping constraints: :math:`\forall (s_i, t_i), (s_j, t_j) \in \mathcal{A}, \quad (s_i \neq s_j) \land (t_i \neq t_j)`

Usage
---------

This module guides you through a step-by-step process for performing ensemble-based ontology alignment using multiple OntoAligner models. By the end, you’ll understand how to configure aligner pipelines, combine their predictions with voting strategies, evaluate the final matchings, and save the outputs in XML and JSON formats.

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
    and ensemble aligners. Each aligner is represented by an :class:`AlignerPipeline`
    and may include aligner-level postprocessing before voting.

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
            )
        ]

    Each aligner is represented as a tuple containing the aligner name, an
    :class:`AlignerPipeline`, and an optional aligner weight.

    .. code-block:: python

        aligners = [
            ("lightweight", AlignerPipeline(...), 1.0),
            ("sbert", AlignerPipeline(...), 1.0),
        ]

    The aligner weight controls how much influence the aligner has during voting.

.. tab:: ➡️ 4: Ensemble Learning Aligner

    Initialize :class:`EnsembleLearningAligner` with the configured aligners and a voting
    method. The default voting method is :class:`ReciprocalRankFusionVoting`.

    .. note::

        Ensemble aligners are executed sequentially in the current implementation: each
        aligner generates its predictions in turn, and the voting step is applied after all
        aligner outputs are collected.

    .. code-block:: python

        ensemble = EnsembleLearningAligner(
            aligners=aligners,
            voting=ReciprocalRankFusionVoting(k=60, selection="top1_source"),
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

Nested Ensemble Learning
----------------------------

Nested ensemble learning extends the standard ensemble learning workflow by combining multiple
ensemble groups into one final ensemble. Instead of placing every aligner pipeline in a
single flat ensemble, related pipelines are first grouped with
:class:`EnsembleLearningAligner`. These group-level ensembles are then combined again
with another :class:`EnsembleLearningAligner`.

This is useful when an alignment workflow uses different groups of signals, such as
retrieval, reranking, graph structure, and LLM-based reasoning.

.. hint::

    Nested ensembles can mix different :class:`AlignerPipeline` configurations and
    ensemble groups as long as they expose the standard ``generate()`` flow. This makes
    it possible to combine pipelines with different encoders, aligners, postprocessors,
    and rerankers in the same workflow.

The nested ensemble follows the flow below:

.. raw:: html

    <div align="center">
        <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/refs/heads/dev/docs/source/img/nested_ensemble.png" width="80%"/>
    </div>

A nested ensemble can be configured by first creating the group-level ensembles and then
passing those ensembles into the final ensemble.

.. code-block:: python

    llm_ensemble = EnsembleLearningAligner(
        aligners=[
            ("llm", llm_pipeline, 1.0),
            ("rag", rag_pipeline, 1.0),
            ("fsrag", fsrag_pipeline, 1.0),
        ],
        voting=ReciprocalRankFusionVoting(k=60),
    )

    retrieval_ensemble = EnsembleLearningAligner(
        aligners=[
            ("lightweight", lightweight_pipeline, 1.0),
            ("tfidf", tfidf_pipeline, 1.0),
            ("sbert", sbert_pipeline, 1.0),
        ],
        voting=ReciprocalRankFusionVoting(k=60),
    )

    reranking_ensemble = EnsembleLearningAligner(
        aligners=[
            ("sbert_reranking", sbert_reranking_pipeline, 1.0),
            ("tfidf_reranking", tfidf_reranking_pipeline, 1.0),
            ("graph_reranking", graph_reranking_pipeline, 1.0),
        ],
        voting=ScoreAverageVoting(),
    )

    nested_ensemble = EnsembleLearningAligner(
        aligners=[
            ("llm_ensemble", llm_ensemble, 1.0),
            ("retrieval_ensemble", retrieval_ensemble, 1.0),
            ("reranking_ensemble", reranking_ensemble, 1.0),
        ],
        voting=ReciprocalRankFusionVoting(k=60),
    )

    final_matchings = nested_ensemble.generate()

.. note::

    A complete tutorial notebook is available at
    `tutorial/04-nested-ensemble-learning-aligners-in-ontoaligner.ipynb <https://github.com/sciknoworg/OntoAligner/blob/dev/tutorial/04-nested-ensemble-learning-aligners-in-ontoaligner.ipynb>`_.

Voting Strategies
-----------------------

Voting strategies combine normalized predictions from multiple aligners. Each aligner
contributes a list of predictions and an aligner weight. The aligner weight controls the
influence of the aligner during fusion.


.. list-table::
   :header-rows: 1
   :widths: 25 50 15

   * - Strategy
     - Description
     - Link
   * - ``ReciprocalRankFusionVoting``
     - Adds reciprocal-rank scores from each aligner and ranks pairs by the fused score.
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
     - Counts weighted aligner support for each source-target pair and filters by vote settings.
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
        voting=ReciprocalRankFusionVoting(k=60, selection="top1_source"),
    )

All voting classes now share the same post-fusion selection controls from
:class:`BaseVoting`: ``selection`` (``none``, ``top1_source``, ``bijective``,
``threshold``, or ``topk_source``), plus ``threshold``, ``top_k``, and ``margin``.
These options let you decide whether the fused output should stay ranked, be
reduced to one match per source, or keep multiple candidates per source.

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
         - A list of aligner tuples in the form ``(name, aligner_pipeline)`` or
           ``(name, aligner_pipeline, weight)``. At least two aligner pipelines
           are required.
       * - **voting**
         - BaseVoting
         - ``ReciprocalRankFusionVoting()``
         - Voting method used to combine aligner predictions.
       * - **selection**
         - str
         - ``"top1_source"`` for most voting methods
         - Post-fusion selection strategy: ``none``, ``top1_source``, ``bijective``,
           ``threshold``, or ``topk_source``.
       * - **threshold**
         - float
         - ``None``
         - Score cutoff used when ``selection="threshold"``.
       * - **top_k**
         - int
         - ``None``
         - Maximum targets kept per source when ``selection="topk_source"``.
       * - **margin**
         - float
         - ``None``
         - Relative score margin used with ``selection="topk_source"``.
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
       * - **selection**
         - str
         - ``"top1_source"``
         - Post-fusion selection strategy applied to the ranked fused output.
       * - **threshold**
         - float
         - ``None``
         - Score cutoff used when ``selection="threshold"``.
       * - **top_k**
         - int
         - ``None``
         - Maximum targets kept per source when ``selection="topk_source"``.
       * - **margin**
         - float
         - ``None``
         - Relative score margin used with ``selection="topk_source"``.

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
         - Minimum aligner score required to count a vote.
       * - **selection**
         - str
         - ``"top1_source"``
         - Post-fusion selection strategy applied after weighted voting.
       * - **threshold**
         - float
         - ``None``
         - Score cutoff used when ``selection="threshold"``.
       * - **top_k**
         - int
         - ``None``
         - Maximum targets kept per source when ``selection="topk_source"``.
       * - **margin**
         - float
         - ``None``
         - Relative score margin used with ``selection="topk_source"``.

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

    BordaCountVoting, CondorcetVoting, and ScoreAverageVoting also accept the same
    shared :class:`BaseVoting` selection controls (``selection``, ``threshold``,
    ``top_k``, and ``margin``) in addition to their own voting-specific settings.

Configuration Example:

.. code-block:: python

    ensemble = EnsembleLearningAligner(
        aligners=aligners,
        voting=ReciprocalRankFusionVoting(k=60, selection="topk_source", top_k=3),
    )

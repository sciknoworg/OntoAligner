Ensemble Learning
===========================

Ensemble Alignment
------------------------

The :mod:`ontoaligner.aligner.ensemble` module provides ensemble learning support for ontology alignment in OntoAligner.

Ensemble learning combines predictions from multiple ontology alignment models to produce a final alignment result. In ontology matching, different aligners may capture different signals, such as lexical similarity, retrieval similarity, graph structure, or LLM-based semantic verification.

The ensemble aligner in OntoAligner provides a common interface for running multiple alignment branches and combining their outputs with a voting method. Each branch can use a different encoder, aligner, and optional postprocessor. The final predictions are normalized into a common ``source``-``target``-``score`` format before voting.

The ensemble aligner follows the standard OntoAligner model flow. Each branch encodes a collected ontology matching dataset, loads the aligner when needed, generates predictions, and optionally applies branch-level postprocessing. The final branch outputs are then combined using a voting strategy.

The module provides two main execution components:

1. :class:`AlignerPipeline` — runs one encoder and one ontology matching aligner.
2. :class:`EnsembleLearningAligner` — runs multiple :class:`AlignerPipeline` objects and combines their predictions.

The module also provides voting strategies for combining branch outputs:

1. :class:`ReciprocalRankFusion`
2. :class:`BordaCountVoting`
3. :class:`CondorcetVoting`
4. :class:`ScoreAverageVoting`
5. :class:`WeightedVoting`

The ensemble output follows the standard OntoAligner alignment format:

.. code-block:: python

    [
        {"source": "source_iri", "target": "target_iri", "score": 0.9},
        ...
    ]

Grouped retrieval outputs such as ``target-cands`` and ``score-cands`` are flattened before voting.

Voting Strategies
-----------------------

The ensemble module supports both rank-based and score-based voting methods.

Rank-based methods are useful when branch scores are not directly comparable. This is common when combining lightweight, retrieval, KGE, LLM, and RAG-based aligners.

Score-based methods are useful when branch outputs come from similar model families or when their scores are already comparable.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Voting Strategy
     - Description
     - Recommended Use
   * - ``ReciprocalRankFusion``
     - Combines ranked branch outputs using reciprocal-rank scores.
     - Heterogeneous ensembles
   * - ``BordaCountVoting``
     - Assigns rank-based points to predictions and sums them across branches.
     - Heterogeneous ranked outputs
   * - ``CondorcetVoting``
     - Compares target candidates pairwise for each source entity.
     - Candidate preference voting
   * - ``ScoreAverageVoting``
     - Combines branch scores using weighted score averaging.
     - Same-family aligners
   * - ``WeightedVoting``
     - Counts weighted branch support for each source-target pair.
     - Agreement-based filtering

Each voting method receives branch predictions together with a branch weight:

.. code-block:: python

    [
       ([{"source": "...", "target": "...", "score": 0.9}], 1.0),
       ([{"source": "...", "target": "...", "score": 0.8}], 1.0),
    ]

The first element in each tuple is the list of normalized predictions from one branch. The second element is the branch weight used during voting.

Usage
----------

.. tab:: ➡️ 1: Import

   Import the ensemble components, voting methods, and standard OntoAligner modules.

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
          ConceptParentFewShotEncoder,
          GraphTripleEncoder,
      )
      from ontoaligner.aligner import (
          SimpleFuzzySMLightweight,
          TFIDFRetrieval,
          SBERTRetrieval,
          AutoModelDecoderLLM,
          ConceptLLMDataset,
          MistralLLMBERTRetrieverRAG,
          MistralLLMBERTRetrieverFSRAG,
          TransEAligner,
      )
      from ontoaligner.postprocess import (
          TFIDFLabelMapper,
          llm_postprocessor,
          graph_postprocessor,
          rag_heuristic_postprocessor,
      )
      from ontoaligner.aligner.ensemble import (
          AlignerBranch,
          EnsembleAligner,
          ReciprocalRankFusion,
          ScoreAverageVoting,
      )

.. tab:: ➡️ 2: Parse Ontologies

   Load the source ontology, target ontology, and optional reference alignment using an OntoAligner dataset class.

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

.. tab:: ➡️ 3: Single Aligner Branch

   :class:`AlignerBranch` can be used independently to run one encoder and one aligner.

   .. code-block:: python

      branch = AlignerBranch(
          encoder=ConceptParentLightweightEncoder(),
          aligner=SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2),
          om_dataset=dataset,
      )

      matchings = branch.generate()

      evaluation = metrics.evaluation_report(
          predicts=matchings,
          references=dataset["reference"],
      )

      print("Single Branch Evaluation Report:")
      print(json.dumps(evaluation, indent=4))

.. tab:: ➡️ 4: Heterogeneous Ensemble

   Use a rank-based voting method when combining aligners from different model families. This example combines lightweight, retrieval, SBERT retrieval, KGE, LLM, RAG, and FewShot-RAG branches.

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

      branches = [
          (
              "lightweight",
              AlignerBranch(
                  encoder=ConceptParentLightweightEncoder(),
                  aligner=SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2),
                  om_dataset=dataset,
              ),
              1.0,
          ),
          (
              "tfidf",
              AlignerBranch(
                  encoder=ConceptParentLightweightEncoder(),
                  aligner=TFIDFRetrieval(top_k=5),
                  om_dataset=dataset,
                  load_params={"path": None},
              ),
              1.0,
          ),
          (
              "sbert",
              AlignerBranch(
                  encoder=ConceptParentLightweightEncoder(),
                  aligner=SBERTRetrieval(device=device, top_k=5),
                  om_dataset=dataset,
                  load_params={"path": ir_model_path},
              ),
              1.0,
          ),
          (
              "kge",
              AlignerBranch(
                  encoder=GraphTripleEncoder(),
                  aligner=TransEAligner(
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
              AlignerBranch(
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
              AlignerBranch(
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
          (
              "fsrag",
              AlignerBranch(
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
              ),
              1.0,
          ),
      ]

      ensemble = EnsembleAligner(
          branches=branches,
          voting=ReciprocalRankFusion(k=60),
      )

      matchings = ensemble.generate()

.. tab:: ➡️ 5: Score-Based Ensemble

   Use :class:`ScoreAverageVoting` when the branches produce comparable scores.

   .. code-block:: python

      device = "cuda" if torch.cuda.is_available() else "cpu"

      branches = [
          (
              "tfidf",
              AlignerBranch(
                  encoder=ConceptParentLightweightEncoder(),
                  aligner=TFIDFRetrieval(top_k=5),
                  om_dataset=dataset,
                  load_params={"path": None},
              ),
              1.0,
          ),
          (
              "sbert",
              AlignerBranch(
                  encoder=ConceptParentLightweightEncoder(),
                  aligner=SBERTRetrieval(device=device, top_k=5),
                  om_dataset=dataset,
                  load_params={"path": "all-MiniLM-L6-v2"},
              ),
              1.0,
          ),
      ]

      ensemble = EnsembleAligner(
          branches=branches,
          voting=ScoreAverageVoting(),
      )

      matchings = ensemble.generate()

.. tab:: ➡️ 6: Evaluate and Export

   Evaluate the ensemble output and export the final alignment to XML.

   .. code-block:: python

      evaluation = metrics.evaluation_report(
          predicts=matchings,
          references=dataset["reference"],
      )

      print("Ensemble Evaluation Report:")
      print(json.dumps(evaluation, indent=4))

      xml_str = xmlify.xml_alignment_generator(matchings=matchings)

      with open("ensemble_matchings.xml", "w", encoding="utf-8") as xml_file:
          xml_file.write(xml_str)


Aligner Pipeline
----------------------

The `AlignerPipeline` runs one encoder and one ontology aligner over a collected ontology matching dataset. It can be used independently or as a branch inside :class:`EnsembleAligner`. The parameters of this module are as follows:


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Type
     - Description
   * - ``encoder``
     - ``BaseEncoder``
     - Encoder used to encode the source and target ontology items.
   * - ``aligner``
     - ``BaseOMModel``
     - Ontology matching aligner used to generate predictions.
   * - ``om_dataset``
     - ``dict``
     - Pre-collected ontology matching dataset.
   * - ``load_params``
     - ``dict``
     - Parameters forwarded to the aligner ``load`` method when provided.
   * - ``llm_dataset_class``
     - ``Dataset``
     - Dataset class used to wrap LLM inputs.
   * - ``batch_size``
     - ``int``
     - Batch size used for LLM prompt generation.
   * - ``shuffle``
     - ``bool``
     - Whether to shuffle LLM batches.
   * - ``postprocessor``
     - ``Any``
     - Optional branch-level postprocessor.
   * - ``postprocessor_params``
     - ``dict``
     - Parameters forwarded to the postprocessor.
   * - ``include_reference``
     - ``bool``
     - Whether to pass reference matchings to the encoder.

.. note::

	``AlignerBranch`` applies postprocessing only when a postprocessor is provided. This is useful for model families that need output conversion before voting, such as LLM, RAG, FewShot-RAG, or KGE models.

EnsembleAligner
--------------------

The ``EnsembleAligner`` combines predictions from two or more aligner branches. Parameters are as follows:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Type
     - Description
   * - ``branches``
     - ``list``
     - A list of branch tuples in the form ``(name, branch)`` or ``(name, branch, weight)``.
   * - ``voting``
     - ``BaseVoting``
     - Voting method used to combine branch predictions. Defaults to :class:`ReciprocalRankFusion`.

.. note::

	``EnsembleAligner`` requires two or more branches. If only one branch is provided, it raises an error.


Before voting, grouped retrieval outputs are flattened into the standard alignment format. Flat predictions with ``source`` and ``target`` are also accepted. If a flat prediction does not contain ``score``, the ensemble assigns a default score of ``1.0`` before voting.

.. tab:: ReciprocalRankFusion

	The `ReciprocalRankFusion` combines ranked pipeline outputs using reciprocal-rank scores. The parameters are as follows:

	.. list-table::
	   :header-rows: 1
	   :widths: 25 20 55

	   * - Parameter
	     - Type
	     - Description
	   * - ``k``
	     - ``int``
	     - Smoothing constant used in reciprocal rank fusion. Defaults to ``60``.

	Use :class:`ReciprocalRankFusion` when aligner pipelines come from different model families or when branch scores are not directly comparable.


.. tab:: BordaCountVoting

	:class:`BordaCountVoting` combines ranked branch outputs by assigning higher scores to higher-ranked predictions.


	Use :class:`BordaCountVoting` when the ranking order of each branch is more important than the raw branch scores.

.. tab::  CondorcetVoting


	The `CondorcetVoting` compares target candidates pairwise for each source entity and ranks candidates by pairwise victories.


	Use :class:`CondorcetVoting` when target candidates should be compared pairwise within each source entity.

.. tab::  ScoreAverageVoting

	:class:`ScoreAverageVoting` combines branch predictions by averaging scores across branches using branch weights.

	Use :class:`ScoreAverageVoting` when branch scores are comparable or already calibrated.

.. tab::  WeightedVoting

	:class:`WeightedVoting` combines branch predictions by counting weighted branch support for each source-target pair. The parameters are as follows:

	.. list-table::
	   :header-rows: 1
	   :widths: 25 20 55

	   * - Parameter
	     - Type
	     - Description
	   * - ``min_votes``
	     - ``int``
	     - Minimum number of branches required for a pair. Defaults to ``1``.
	   * - ``score_threshold``
	     - ``float``
	     - Minimum branch score required to count a vote. Defaults to ``None``.

	Use :class:`WeightedVoting` when the ensemble should favor predictions supported by one or more branches. With equal branch weights, it behaves like majority-style voting.

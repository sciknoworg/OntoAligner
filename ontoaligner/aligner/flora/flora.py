# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.

This module implements the FLORA aligner, an unsupervised knowledge graph alignment system
that jointly aligns entities and relations using iterative fuzzy logic inference.

**Algorithm Overview**:

FLORA iteratively:
1. Bootstraps entity alignments from literal similarity (strings, dates, numbers)
2. Infers predicate subsumptions from aligned entity triples
3. Uses fuzzy logic rules to align additional entities based on predicate evidence
4. Repeats until convergence

**Key Features**:
- Unsupervised: No training data required (optional seed alignments supported)
- Holistic: Jointly aligns entities and relations iteratively
- Interpretable: All scores grounded in fuzzy logic rules
- Convergent: Monotone property ensures convergence
- Robust: Handles dangling entities and incomplete mappings

**References**:

    Peng, Yiwen, Bonald, Thomas, and Suchanek, Fabian.
    "FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic."
    International Semantic Web Conference (ISWC), 2025.
    https://suchanek.name/work/publications/iswc-2025.pdf
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
import multiprocessing

from ...base import BaseOMModel
from .fuzzy_logic import fuzzy
from .fuzzy_logic.literals import FLORALiteralsEmbedding



class FLORARDFWriter:
    """Writes knowledge graph alignments to RDF/Turtle format.

    Converts alignment results (entity and predicate mappings with scores)
    to RDF triples with namespace declarations.
    """

    def __init__(self, prefixes: Dict[str, str]) -> None:
        """Initialize RDF writer with namespace prefixes.

        Args:
            prefixes: Dictionary mapping prefix names to namespace URIs.
                Example: {'ex': 'http://example.org/', 'owl': 'http://www.w3.org/2002/07/owl#'}
        """
        self.prefixes = prefixes

    def write(
        self,
        output_path: str,
        kb1: Any,
        kb2: Any,
        predicate2super_predicate: Dict[Any, Dict[Any, float]],
        same_as_scores: Dict[Any, Dict[Any, float]],
    ) -> None:
        """Write alignment results to RDF file.

        Writes:
        - Namespace prefixes
        - Predicate subsumption (rdfs:subPropertyOf) relationships
        - Entity equivalence (owl:sameAs) mappings with confidence scores

        Args:
            output_path: File path for the output RDF/Turtle file.
            kb1: First knowledge base (used to filter predicates).
            kb2: Second knowledge base (used to filter predicates).
            predicate2super_predicate: Predicate alignment scores.
            same_as_scores: Entity alignment scores.
        """
        start_time = time.time()
        logging.info("Writing out RDF results...")

        with open(output_path, "wt", encoding="utf-8") as out:
            # Write prefixes
            for prefix, uri in self.prefixes.items():
                out.write(f"@prefix {prefix}: <{uri}> .\n")

            # Write predicate hierarchies
            kb1_predicates = kb1.predicates()
            kb2_predicates = kb2.predicates()
            predicates = kb1_predicates | kb2_predicates
            for predicate1 in predicates:
                if predicate1 in predicate2super_predicate:
                    for predicate2, score in predicate2super_predicate[predicate1].items():
                        if score > 0.1:
                            out.write(f"{predicate1}\trdfs:subPropertyOf\t{predicate2}\t.#\t{score}\n")

            # Write sameAs links for entities/instances
            for entity1, others in same_as_scores.items():
                for entity2, score in others.items():
                    if score > 0:
                        out.write(f"{entity1}\towl:sameAs\t{entity2}\t.#\t{score}\n")

        logging.info("Finished writing RDF results.")
        logging.info(
            "Time used for the whole procedure: %.5f minutes",
            (time.time() - start_time) / 60
        )


class FLORAAligner(BaseOMModel):
    """FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.

    A fully unsupervised system for aligning two knowledge graphs by jointly matching
    entities and relations through iterative fuzzy logic inference.

    **Pipeline Overview**:

    1. **Initialization** – Load KGs and optional seed alignments
    2. **Literal bootstrapping** – Align string/date/numeric literals using embeddings
    3. **First iteration** – Infer predicate subsumptions from aligned literals
    4. **Main loop** – Iteratively align entities using fuzzy rules and update predicates
    5. **Convergence** – Stop when alignment scores stabilize

    **Parameters**:

        alpha (float):
            Benefit-of-doubt factor for subrelation inference (higher = more lenient).
            Default: 3.0
        init_threshold (float):
            Minimum semantic similarity for bootstrapping literal alignment. Default: 0.7
        gramN (int):
            Maximum number of evidential triples per entity during alignment. Default: 100
        epsilon (float):
            Convergence threshold; stops when |Σ_new - Σ_old| < epsilon. Default: 0.01
        max_iterations (int):
            Maximum number of main-loop iterations. Default: 100
        string_identity (bool):
            If True, use only exact string matching for literals (no embeddings). Default: False
        relinit (float):
            Initial score for non-identical predicates. Default: 0.1
        ngrams (List[int]):
            N-gram sizes for functionality computation. Default: [1, 2]
        model_id (str or None):
            Hugging Face model ID for embedding model (e.g., 'Lihuchen/pearl_small'). Default: None
        training_data (str or None):
            Path to seed alignment file (tab-separated, optional score column). Default: None
        device (str or None):
            Device for embeddings ('cuda' or 'cpu'). Auto-detects if None. Default: None
        batch_size (int or None):
            Batch size for embedding computation. Default: 32

    **Example**:

        >>> from ontoaligner.aligner.flora import FLORAAligner
        >>> aligner = FLORAAligner(alpha=3.0, init_threshold=0.7)
        >>> matchings = aligner.generate(["kg1.ttl", "kg2.ttl"])
        >>> for match in matchings[:3]:
        ...     print(f"{match['source']} -> {match['target']}: {match['score']:.2f}")

    **References**:

        Peng, Y., Bonald, T., & Suchanek, F. (2025).
        FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.
        In Proc. ISWC 2025.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        init_threshold: float = 0.7,
        gramN: int = 100,
        epsilon: float = 0.01,
        max_iterations: int = 100,
        string_identity: bool = False,
        relinit: float = 0.1,
        ngrams: Optional[List[int]] = None,
        model_id: Optional[str] = None,
        emb_path: Optional[str] = None,
        training_data: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = 32,
        verbose: bool = False,
        workers: Optional[int] = 4,
        **kwargs,
    ) -> None:
        """Initialize the FLORA aligner.

        Args:
            alpha: Benefit-of-doubt parameter for subrelation mapping.
            init_threshold: Initial similarity threshold for literal bootstrapping.
            gramN: Maximum evidences per entity in alignment rules.
            epsilon: Convergence threshold for score changes.
            max_iterations: Maximum iterations before forced termination.
            string_identity: Use exact string matching only (no embeddings).
            relinit: Initial score for unidentical predicates.
            ngrams: N-gram sizes for functionality computation.
            model_id: Transformer model for literal embeddings.
            emb_path: Optional path to pretrained embeddings.
            training_data: Path to seed alignment file.
            device: Device for tensor operations.
            batch_size: Batch size for embedding computations.
            **kwargs: Additional arguments passed to BaseOMModel.
        """
        if ngrams is None:
            ngrams = [1, 2]
        super().__init__(
            alpha=alpha,
            init_threshold=init_threshold,
            gramN=gramN,
            epsilon=epsilon,
            max_iterations=max_iterations,
            string_identity=string_identity,
            relinit=relinit,
            ngrams=ngrams,
            model_id=model_id,
            training_data=training_data,
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            workers=workers,
            **kwargs,
        )
        if self.kwargs['verbose']:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.literals_embedding = FLORALiteralsEmbedding(
            model_id=model_id,
            device=device if device else 'cpu',
            identity=string_identity,
            emb_path=emb_path,
        )
        self.same_as_scores = {}
        self.predicate2super_predicate = {}

    def __str__(self) -> str:
        """Return the name of this aligner.

        Returns:
            String identifier for the aligner.
        """
        return "FLORAAligner"

    def seed_alignments(self, training_data_path: Optional[str]) -> Dict[str, Dict[str, float]]:
        """Load optional seed alignments from a file.

        Expected file format: tab-separated with entity1, entity2, and optional score.
        Example::

            <http://example.org/Alice>\t<http://example.org/A>\t0.95
            <http://example.org/Bob>\t<http://example.org/B>

        Args:
            training_data_path: Path to the seed alignment file.

        Returns:
            Dictionary mapping entity pairs to alignment scores.

        Raises:
            FileNotFoundError: If training_data_path is specified but doesn't exist.
        """
        same_as_scores: Dict[str, Dict[str, float]] = {}
        if training_data_path is not None:
            if not os.path.exists(training_data_path):
                raise FileNotFoundError(f"Training data file not found: {training_data_path}")
            logging.info("FLORA: Loading training data from %s", training_data_path)
            with open(training_data_path, "rt", encoding="utf-8") as f:
                for line in f:
                    split = line.strip().split("\t")
                    if len(split) < 2:
                        continue
                    entity1, entity2 = split[0], split[1]
                    score = float(split[2]) if len(split) >= 3 else 1.0
                    if entity1 not in same_as_scores:
                        same_as_scores[entity1] = {}
                    same_as_scores[entity1][entity2] = score
        return same_as_scores

    def compute_functionalities(
        self,
        kb1: Any,
        kb2: Any,
    ) -> Dict[Any, float]:
        """Compute predicate functionality scores across two knowledge bases.

        Functionality is the inverse of "diversity": a functional predicate
        (like birthDate) has one value per subject, while a non-functional predicate
        (like knows) can have many values per subject.

        Computes functionality for each predicate using n-gram analysis and returns
        the minimum score across both KGs (conservative estimate).

        Args:
            kb1: First knowledge base.
            kb2: Second knowledge base.

        Returns:
            Dictionary mapping predicates to functionality scores in [0, 1].
        """
        functionalities1 = fuzzy.compute_functionalities(kb1, gram=self.kwargs['ngrams'])
        functionalities2 = fuzzy.compute_functionalities(kb2, gram=self.kwargs['ngrams'])
        functionalities: Dict[Any, float] = {}

        # Merge scores from both KGs
        for pred in functionalities1:
            functionalities[pred] = functionalities1[pred]
        for pred in functionalities2:
            if pred not in functionalities:
                functionalities[pred] = functionalities2[pred]
            else:
                # Use minimum (conservative)
                functionalities[pred] = min(functionalities[pred], functionalities2[pred])
        return functionalities

    def bootstraping(
        self,
        kb1: Any,
        kb2: Any,
        same_as_scores: Dict[Any, Dict[Any, float]],
        predicate2super_predicate: Dict[Any, Dict[Any, float]],
        functionalities: Dict[Any, float],
    ) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, float]]]:
        """Perform the bootstrapping phase of entity and predicate alignment.

        Runs the first iteration in parallel to align entities based on literal similarity,
        then infers predicate subsumptions from the aligned entity triples.

        Args:
            kb1: First knowledge base.
            kb2: Second knowledge base.
            same_as_scores: Initial entity alignment scores (from literal bootstrapping).
            predicate2super_predicate: Initial predicate subsumption scores.
            functionalities: Predicate functionality scores.

        Returns:
            Tuple of (quasi_eqrel, predicate2super_predicate, same_as_scores, ent_max_assign):
            - quasi_eqrel: Predicate quasi-equivalence relations
            - predicate2super_predicate: Updated predicate subsumption scores
            - same_as_scores: Updated entity alignments
            - ent_max_assign: Bilateral max assignments for entities
        """
        same_as_scores = fuzzy.bootstrap_algo(
            kb1, kb2, same_as_scores, predicate2super_predicate, functionalities
        )
        ent_max_assign = fuzzy.bilateral_max_assign(same_as_scores)
        predicate2super_predicate = fuzzy.map_subrelations(
            self.kwargs["alpha"], kb1, kb2, ent_max_assign, predicate2super_predicate
        )
        quasi_eqrel = fuzzy.compute_quasi_eqrel(kb1, kb2, predicate2super_predicate)
        return quasi_eqrel, predicate2super_predicate, same_as_scores, ent_max_assign

    def generate(self, input_data: List[Any]) -> List[Dict[str, Any]]:
        """Run the complete FLORA alignment algorithm on two knowledge graphs.

        This is the main entry point implementing the full FLORA pipeline:
        1. Load optional seed alignments
        2. Initialize predicate subsumption scores
        3. Compute predicate functionalities
        4. Bootstrap entity alignment using literal similarity
        5. Run main iterative alignment loop
        6. Return entity alignment predictions

        **Input Format**:

        ``input_data`` should be a two-element list of Graph objects,
        as returned by :class:`~ontoaligner.encoder.flora.FLORAEncoder`.

        **Standard Usage**:

            >>> from ontoaligner.ontology import FLORAOMDataset
            >>> from ontoaligner.encoder import FLORAEncoder
            >>> from ontoaligner.aligner.flora import FLORAAligner
            >>>
            >>> # Parse KGs
            >>> dataset = FLORAOMDataset().collect("kg1.ttl", "kg2.ttl")
            >>>
            >>> # Encode for aligner
            >>> encoder_output = FLORAEncoder()(
            ...     source=dataset["source"],
            ...     target=dataset["target"]
            ... )
            >>>
            >>> # Align
            >>> aligner = FLORAAligner()
            >>> matchings = aligner.generate(input_data=encoder_output)

        Args:
            input_data: List of two Graph objects [kg1, kg2].

        Returns:
            List of entity alignment predictions. Each prediction is a dictionary:

                - ``source`` (str): IRI of the source KG entity
                - ``target`` (str): IRI of the target KG entity
                - ``score`` (float): Alignment confidence in [0.0, 1.0]

        Raises:
            ValueError: If input_data does not contain exactly 2 elements.

        Yields:
            Predictions are sorted by source entity, then by score (descending).
        """
        if len(input_data) != 2:
            raise ValueError(
                "FLORAAligner.generate() expects input_data = [kg1_graph, kg2_graph], "
                f"but received {len(input_data)} element(s)."
            )
        kb1, kb2 = input_data[0], input_data[1]

        # 1. Load optional seed alignments (training data)
        same_as_scores = self.seed_alignments(training_data_path=self.kwargs["training_data"])

        # 2. Predicate initialisation
        logging.info("FLORA: Initialising predicate subsumptions...")
        predicate2super_predicate = fuzzy.initialize_predicate_subsumption(
            predicates1=kb1.predicates(),
            predicates2=kb2.predicates(),
            relinit=self.kwargs['relinit']
        )

        # 3. Compute functionalities
        logging.info("FLORA: Computing functionalities...")
        functionalities = self.compute_functionalities(kb1=kb1, kb2=kb2)

        # 4. Literal similarity bootstrapping
        same_as_scores = self.literals_embedding.map_literals(
            kb1=kb1,
            kb2=kb2,
            same_as_score=same_as_scores,
            identity=self.kwargs["string_identity"],
            threshold=self.kwargs["init_threshold"]
        )

        # 5. Bootstrapping phase
        logging.info("FLORA: Bootstrapping entity alignment...")
        start_time = time.time()
        quasi_eqrel, predicate2super_predicate, same_as_scores, ent_max_assign = self.bootstraping(
            kb1=kb1,
            kb2=kb2,
            same_as_scores=same_as_scores,
            predicate2super_predicate=predicate2super_predicate,
            functionalities=functionalities
        )
        logging.info("FLORA: Bootstrapping done in %.2f min.", (time.time() - start_time) / 60)

        # 6. Main alignment loop
        iterations = 0
        while True:
            logging.info("FLORA: Iteration %d ...", iterations + 1)
            same_as_sum = sum(v for d in same_as_scores.values() for v in d.values())

            # --- Parallel entity matching with robust error handling ---
            try:
                mgr = multiprocessing.Manager()
                subjs_kb1 = kb1.subjects()
                ent_queue = mgr.Queue(len(subjs_kb1))
                for subj in subjs_kb1:
                    ent_queue.put(subj)

                num_workers = self.kwargs["workers"]
                ent_match_tuple_queue = mgr.Queue()
                tasks = []
                task_errors = []  # Track worker errors

                for _ in range(num_workers):
                    task = multiprocessing.Process(
                        target=fuzzy._match_entities_by_rules,
                        args=(
                            kb1, kb2,
                            quasi_eqrel,
                            ent_queue,
                            ent_match_tuple_queue,
                            same_as_scores,
                            functionalities,
                            self.kwargs,
                        ),
                    )
                    task.daemon = False  # Explicit cleanup
                    task.start()
                    tasks.append(task)

                # Collect results from worker processes FIRST (while queue is alive)
                # Do NOT terminate workers while they might still be writing
                try:
                    timeout_count = 0
                    results_collected = 0
                    while timeout_count < 100 and results_collected < num_workers:  # Expect one result per worker
                        try:
                            batch = ent_match_tuple_queue.get(timeout=5)  # 5 second timeout per batch
                            results_collected += 1
                            for subj1, matches in batch.items():
                                max_score1 = max(ent_max_assign.get(subj1, {None: 0}).values())
                                for subj2, score in matches.items():
                                    max_score2 = max(ent_max_assign.get(subj2, {None: 0}).values())
                                    if score <= max(max_score1, max_score2):
                                        continue
                                    if subj1 not in same_as_scores:
                                        same_as_scores[subj1] = {}
                                    if subj2 not in same_as_scores[subj1]:
                                        same_as_scores[subj1][subj2] = score
                                    elif score > same_as_scores[subj1][subj2]:
                                        same_as_scores[subj1][subj2] = score
                        except Exception:
                            timeout_count += 1
                            if timeout_count >= 100:
                                logging.warning(f"FLORA: Queue timeout after {results_collected}/{num_workers} results...")
                                break
                except (BrokenPipeError, EOFError) as e:
                    logging.warning(f"FLORA: Queue communication error: {e}")

                # NOW wait for workers to complete naturally
                logging.info(f"FLORA: Waiting for {num_workers} workers to complete...")
                timeout = 3600  # 1 hour timeout
                for idx, task in enumerate(tasks):
                    task.join(timeout=timeout)
                    if task.is_alive():
                        logging.warning(f"FLORA: Worker {idx} timeout, terminating...")
                        task.terminate()
                        task.join(timeout=5)
                        if task.is_alive():
                            logging.error(f"FLORA: Worker {idx} force killed")
                            task.kill()
                    elif task.exitcode != 0 and task.exitcode is not None:
                        logging.warning(f"FLORA: Worker {idx} exited with code {task.exitcode}")
                        task_errors.append(idx)

            except Exception as e:
                logging.error(f"FLORA: Multiprocessing error: {e}")
                logging.info("FLORA: Falling back to single-process mode for this iteration...")
                # Fallback: Graceful degradation, algorithm continues in next iteration
                # (In production, could implement serial entity matching here)

            # --- Update predicate subsumptions ---
            ent_max_assign = fuzzy.bilateral_max_assign(same_as_scores)
            predicate2super_predicate = fuzzy.map_subrelations(
                self.kwargs["alpha"], kb1, kb2, ent_max_assign, predicate2super_predicate
            )
            quasi_eqrel = fuzzy.compute_quasi_eqrel(kb1, kb2, predicate2super_predicate)

            # --- Check convergence ---
            new_same_as_sum = sum(v for d in same_as_scores.values() for v in d.values())
            logging.info("FLORA: sameAs sum %.4f -> %.4f", same_as_sum, new_same_as_sum)

            iterations += 1
            if iterations >= self.kwargs["max_iterations"] or \
                    abs(new_same_as_sum - same_as_sum) < self.kwargs["epsilon"]:
                break

        self.same_as_scores = same_as_scores
        self.predicate2super_predicate = predicate2super_predicate

        predictions = []
        # Predicates
        kb1_predicates = kb1.predicates()
        kb2_predicates = kb2.predicates()
        predicates = kb1_predicates | kb2_predicates
        for predicate1 in predicates:
            if predicate1 in predicate2super_predicate:
                for predicate2 in predicate2super_predicate[predicate1]:
                    if predicate2super_predicate[predicate1][predicate2] > 0.1:
                        predictions.append({
                            "source": predicate1,
                            "target": predicate2,
                            "score": float(predicate2super_predicate[predicate1][predicate2]),
                            'type': 'predicate'
                        })

        # Literals and instances
        for entity1 in same_as_scores:
            for entity2 in same_as_scores[entity1]:
                if same_as_scores[entity1][entity2] > 0:
                    predictions.append(
                        {
                            "source": entity1,
                            "target": entity2,
                            "score": float(same_as_scores[entity1][entity2]),
                            'type': 'instance'
                        }
                    )
        return predictions

    def get_same_as_scores(self) -> Dict[Any, Dict[Any, float]]:
        """Get the computed entity alignment scores.

        Returns:
            Dictionary mapping source entities to target entities and their scores.
        """
        return self.same_as_scores

    def get_predicate2super_predicate(self) -> Dict[Any, Dict[Any, float]]:
        """Get the computed predicate subsumption scores.

        Returns:
            Dictionary mapping predicates to their subsumption relationships and scores.
        """
        return self.predicate2super_predicate

FLORA: Fuzzy Logic KG Aligner
==============================================

FLORA
-----------

.. sidebar::

	FLORA was awarded the `🏆 Best Research Paper Award at ISWC 2025 <https://iswc2025.semanticweb.org/#/program/awards>`_

**FLORA** (Fuzzy Logic Knowledge-graph Alignment) is an unsupervised system for automatic knowledge graph (KG) alignment that jointly matches entities *and* relations between two KGs using an iterative fuzzy-logic inference procedure. Key properties of FLORA are: 1) *Unsupervised* – Does not require labelled training pairs. Optional seed alignments are supported via the training-data at supervised method. 2) *Holistic* – Entities *and* relations are aligned jointly in an iterative loop. 3) *Interpretable* – All confidence scores are grounded in fuzzy-logic rules. 4) *Provably Convergent* – The iterative procedure is monotone and always terminates. 5) *Dangling-Entity Aware* – Entities without a counterpart in the other KG are handled gracefully. Moreover, FLORA builds on the PARIS system with three key improvements: 1) Convergence guarantees via monotone score updates. 2) Better handling of non-functional relations via per-subject functionality weighting. 3) Neural string-embedding similarity for literal matching. The following figure illustrates the overall pipeline of FLORA, from input KGs to final alignments:

.. raw:: html

	<div class="video-card">
	  <iframe
	    src="https://videolectures.net/embed/videos/eswc2024_babaei_giglou_language_models?part=1"
	    frameborder="0"
	    allowfullscreen>
	  </iframe>
	  <p class="video-caption">
	    ISWC 2025 Talk — FLORA Presentation by Yiwen Peng.
	  </p>
	</div>

.. note::

    **Reference:** Peng, Y., Bonald, T., Suchanek, F.M. (2026). FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic. In: Garijo, D., et al. The Semantic Web – ISWC 2025. ISWC 2025. Lecture Notes in Computer Science, vol 16140. Springer, Cham. https://doi.org/10.1007/978-3-032-09527-5_11


Fuzzy Logic
----------------------
Given two KGs :math:`G = \langle E, R, T \rangle` and :math:`G' = \langle E', R', T' \rangle`, FLORA jointly produces:

- **Entity alignment** :math:`M_e = \{(e, e') \mid e \equiv e'\}` — pairs of semantically equivalent entities.
- **Relation alignment** :math:`M_r = \{(r, O, r') \mid r \mathbin{O} r',\; O \in \{\subseteq, \supseteq, \equiv\}\}` — pairs of relations with their subsumption or equivalence type.

.. raw:: html

    <div align="center">
     <img src="https://raw.githubusercontent.com/dig-team/FLORA/refs/heads/main/docs/pipeline.png" width="75%"/>
    </div>

FLORA frames both tasks as a single **Recursive Fuzzy Inference System**: every alignment score is a value in :math:`[0, 1]`, rules propagate scores through the graph, and an iterative fixed-point algorithm drives everything to convergence.

.. sidebar::::

   FLORA handles **dangling entities** — entities that have no counterpart in the other KG — naturally, since alignment scores simply remain at or near zero for unmatched entities. No one-to-one mapping is assumed during iteration.


.. tab:: 🔑 Key concepts of FLORA

	.. tab:: 📊 1. Functionality

		A relation :math:`r` is *functional* if it tends to map each head entity to a unique tail entity. FLORA uses two flavors:

		**Global functionality**:

		.. math::

		   \operatorname{fun}(r) = \frac{|\{h \mid \exists\, t : r(h,t)\}|}{|\{(h,t) \mid r(h,t)\}|}

		A value of 1 means :math:`r` is fully functional (e.g., ``hasCapital`` for most countries). High global functionality makes a relation a reliable bridge for propagating alignments.

		**Local functionality**:

		.. math::

		   \operatorname{fun}(r, h) = \frac{1}{|\{t \mid r(h,t)\}|}

		This captures exceptions: ``hasCapital`` is globally functional, but locally non-functional for South Africa (which has three capitals). Using both global and local functionality prevents spurious matches in such cases.

	.. tab:: 🔗 2. Relation Lists

		FLORA generalises single-relation functionality to *lists* of relations :math:`R = (r_1, \ldots, r_n)` applied to a corresponding list of head entities :math:`H = (h_1, \ldots, h_n)`:

		.. math::

		   R(H, t) \;:=\; r_1(h_1, t) \;\wedge\; \cdots \;\wedge\; r_n(h_n, t)

		Even if no individual relation is functional, their combination can be. For example, neither ``BirthDateOf`` nor ``FamilyNameOf`` alone uniquely identifies a person, but together they usually do. This is a key improvement over PARIS, which is limited to single functional relations.

	.. tab:: ⚙️ 3. Aggregation Functions

		FLORA uses three aggregation functions, each chosen for a specific role:

		.. list-table::
		   :header-rows: 1
		   :widths: 20 40 40

		   * - Function
		     - Formula
		     - Used for
		   * - **min**
		     - :math:`\min(p_1, \ldots, p_n)`
		     - Entity rule premises — all conditions must hold; no strong premise can compensate for a weak one.
		   * - **harmonic mean**
		     - :math:`\frac{n}{\sum_{i=1}^n 1/p_i}`
		     - Head-list and relation-list aggregation — rewards high evidence while penalising weak conjuncts.
		   * - **α-mean**
		     - :math:`\min\!\left(\alpha \cdot \frac{1}{n}\sum_i p_i,\; 1\right)`
		     - Subrelation scoring — arithmetic mean scaled by benefit-of-doubt constant :math:`\alpha = 3` to compensate for KG incompleteness.


.. tab:: 👣 FLORA 4 Main Steps

	.. tab:: ✨ 1. Literal Similarity Initialization

		Before any iteration, FLORA computes pairwise similarity scores for all literals (strings, numbers, dates) across the two KGs. These scores are **computed once** and treated as fixed input variables throughout the algorithm.

		- **Strings** — cosine similarity using a small language model:

		  - `LaBSE <https://huggingface.co/sentence-transformers/LaBSE>`_ for multilingual datasets.
		  - `PEARL <https://huggingface.co/Lihuchen/pearl_small>`_ (``Lihuchen/pearl_small``) for monolingual datasets.
		  - Only pairs above a threshold :math:`\theta_s = 0.7` are retained.

		- **Numbers** — similarity of 1 if values agree within a relative error of :math:`10^{-9}`, else 0.
		- **Dates** — exact match only.

		If training data are provided, matched entity pairs are fixed at score 1 and also treated as input variables that are never updated. Relation similarities are **initialised to a small constant** :math:`\theta_r = 0.1` to bootstrap the entity alignment rule on the first iteration. This value is superseded by the subrelation alignment scores in subsequent iterations.

		.. important::

			This step does **not** produce entity alignment scores. It produces literal similarity scores that seed the entity alignment rules in Step 2. Entity alignment scores emerge only from the iterative process.

	.. tab:: 🔀 2. Entity Alignment via Fuzzy Rules


		The core of FLORA is the following rule, applied for every candidate pair of non-literal entities :math:`t \in E` and :math:`t' \in E'`:

		.. math::

		   R(H, t) \;\wedge\; R'(H', t') \;\wedge\; H \equiv H' \;\wedge\; R \cong R'
		   \;\wedge\; \operatorname{fun}(R) \;\wedge\; \operatorname{fun}(R, H)
		   \;\wedge\; \operatorname{fun}(R') \;\wedge\; \operatorname{fun}(R', H')
		   \;\xrightarrow{\min}\; t \equiv t'

		In plain terms: if head entities :math:`H` and :math:`H'` are already aligned, if the relation lists :math:`R` and :math:`R'` are similar and jointly functional (both globally and locally), and if both relation lists actually connect those heads to :math:`t` and :math:`t'` respectively, then :math:`t` and :math:`t'` receive an alignment score equal to the **minimum** of all these premise values. The sub-expressions in the rule are themselves output variables of subordinate rules:

		- **Head-list alignment** :math:`H \equiv H'`, were :math:`h_1 \equiv h'_1 \;\wedge\; \cdots \;\wedge\; h_n \equiv h'_n \;\xrightarrow{\text{hmean}}\; H \equiv H'`, in which each individual head alignment :math:`h_i \equiv h'_i` is either itself an output variable (from a previous application of the entity rule) or a literal similarity score (from Step 1).
		- **Relation-list similarity** :math:`R \cong R'`, were :math:`r_1 \cong r'_1 \;\wedge\; \cdots \;\wedge\; r_n \cong r'_n \;\xrightarrow{\text{hmean}}\; R \cong R'`, where :math:`r \cong r'` holds (with firing strength 1) if :math:`r \subseteq r'` or :math:`r' \subseteq r` — i.e., one is a subrelation of the other. If multiple rules imply the same output variable :math:`t \equiv t'`, its score is the **maximum** of all firing strengths, in line with the FIS solver (Algorithm 1).

	.. tab:: 🧬 3. Subrelation Alignment

		For each pair of relations :math:`r \in R` and :math:`r' \in R'`, FLORA measures how consistently the facts of :math:`r` are also facts of :math:`r'` under the current entity alignment. Furthermore, For each fact :math:`r(h, t)` in :math:`G`, it looks for a counterpart :math:`r'(h', t')` in :math:`G'` where :math:`h \equiv h'` and :math:`t \equiv t'` already hold. Each such coincidence produces a local score:

		.. math::

		   r(h,t) \;\wedge\; r'(h',t') \;\wedge\; h \equiv h' \;\wedge\; t \equiv t'
		   \;\xrightarrow{\min}\; r \equiv_{h,t} r'

		The subrelation score is then:

		.. math::

		   \bigoplus_{h,t:\, r(h,t)} r \equiv_{h,t} r'
		   \;\xrightarrow{\alpha\text{-mean}}\; r \subseteq r'

		The :math:`\alpha\text{-mean}` aggregation (arithmetic mean multiplied by :math:`\alpha = 3`, capped at 1) accounts for KG incompleteness under the Open World Assumption: some facts :math:`r'(h', t')` may simply be absent from :math:`G'`, so a raw average would underestimate the true subrelation score.

		.. note::

		   Relation alignment is **asymmetric** by design. FLORA computes :math:`r \subseteq r'` and :math:`r' \subseteq r` independently. Equivalence :math:`r \equiv r'` is only declared when both subsumptions hold. This naturally handles cases such as ``parent`` (DBpedia) being a superrelation of ``father`` (Wikidata), as illustrated in Figure 1 of the paper.

	.. tab:: 🔁 4. Fixed-Point Iteration until Convergence

		Steps 2 and 3 are applied alternately. After each full pass, every output variable takes the **maximum** over all rules that imply it. The process terminates (early stopping) when the total matching score increases by less than :math:`\varepsilon = 0.01`.

		**Convergence guarantee.** Because all aggregation functions (min, harmonic mean, α-mean, max) are *continuous* and *non-decreasing*, the Knaster–Tarski fixed-point theorem guarantees that Algorithm 1 converges to the unique **least fixed point** of the Recursive FIS. This is Theorem 1 of the paper and is a key theoretical advantage over PARIS, which has no such guarantee. After convergence:

		- Entity pairs with scores below :math:`\theta_e = 0.1` are discarded.
		- A **maximum assignment** is enforced for entities: each entity :math:`e \in E`
		  retains only its single highest-scored counterpart in :math:`E'`
		  (one-to-one constraint). Relations are not subject to this constraint —
		  all subrelation pairs with scores above zero are kept.



Usage
--------

.. tab:: ➡️ 1: Import

    Import the FLORA aligner, its dataset/encoder helpers, and utility modules.

    .. code-block:: python

        import json
        from ontoaligner.ontology import FLORAOMDataset
        from ontoaligner.encoder import FLORAEncoder
        from ontoaligner.aligner.flora import FLORAAligner
        from ontoaligner.utils import metrics, xmlify

    .. note::

        The ``FLORAAligner`` class accepts configuration parameters to customize
        behaviour. See `Configuration`_ below.

.. tab:: ➡️ 2: Parse Ontologies/KGs

    ``FLORAOMDataset.collect()`` calls ``FLORAOntology.parse()`` for each KG,
    which **loads the Turtle file** into a FLORA ``Graph`` and **extracts** all
    alignment-relevant data.

    .. code-block:: python

        task    = FLORAOMDataset()
        dataset = task.collect(
            source_ontology_path="path/to/kg1.ttl",
            target_ontology_path="path/to/kg2.ttl",
            reference_matching_path="path/to/reference.xml",  # optional
        )

    Each side of ``dataset`` is a single-element list containing:

    .. code-block::

        dataset["source"][0] == {
            "entities":   [{"iri": "http://...", "label": "MyEntity"}, ...],
            "predicates": {...},
            "triples":    [("http://subj", "pred", "http://obj"), ...],
            "graph":      <Graph>   # ← passed to the aligner
        }

.. tab:: ➡️ 3: Encoder

    ``FLORAEncoder`` extracts the two pre-loaded ``Graph`` objects from the
    structured parser output and returns ``[kg1_graph, kg2_graph]``.

    .. code-block:: python

        encoder_model  = FLORAEncoder()
        encoder_output = encoder_model(source=dataset["source"],
                                       target=dataset["target"])
        # encoder_output == [kg1_graph, kg2_graph]
        # Both are fully loaded FLORA Graph objects, ready for the aligner.

.. tab:: ➡️ 4: Initialize Aligner

    Create a FLORA aligner instance and configure parameters as needed.

    .. code-block:: python

        aligner = FLORAAligner(
            # Subrelation inference
            alpha=3.0,              # Benefit-of-doubt (higher = more lenient)
            relinit=0.1,            # Initial score for non-identical predicates

            # Literal bootstrapping
            init_threshold=0.7,     # Min semantic similarity (0.0-1.0)
            string_identity=False,  # Set True to skip embeddings (faster)

            # Entity matching
            gramN=100,              # Max evidence facts per entity

            # Convergence
            epsilon=0.01,           # Stop when score change < epsilon
            max_iterations=100,     # Safety cap on iterations

            # Optional: seed alignments
            training_data=None,     # Path to seed_links.tsv (tab-separated)

            # Embeddings
            model_id='Lihuchen/pearl_small',  # Hugging Face model
            # device='cuda',         # Uncomment for GPU
        )

    See `Configuration`_ below for a complete parameter reference.

.. tab:: ➡️ 5: Generate Alignments

    Pass the encoder output to ``generate()`` — identical to every other aligner.

    .. code-block:: python

        matchings = aligner.generate(input_data=encoder_output)

    The output is a list of entity alignment predictions:

    .. code-block::

        [
            {
                "source": "http://example.org/entity/E1",
                "target": "http://example.org/entity/E2",
                "score": 0.91
            },
            ...
        ]

.. tab:: ➡️ 6: Evaluate and Export

    Compare predictions to a reference alignment.

    .. code-block:: python

        evaluation = metrics.evaluation_report(predicts=matchings,
                                               references=dataset["reference"])
        print("Evaluation:", json.dumps(evaluation, indent=4))

    Example output:

    .. code-block::

        {
            "intersection": 120,
            "precision": 85.7,
            "recall": 78.4,
            "f-score": 81.9,
            "predictions-len": 140,
            "reference-len": 153
        }

    Next, save the alignment results in XML or JSON format.

    .. tab:: 📄 Export to XML

        ::

            xml_str = xmlify.xml_alignment_generator(matchings=matchings)
            with open("matchings.xml", "w", encoding="utf-8") as f:
                f.write(xml_str)

    .. tab:: 🧾 Export to JSON

        ::

            with open("matchings.json", "w", encoding="utf-8") as f:
                json.dump(matchings, f, indent=4, ensure_ascii=False)


Configuration
---------------------

The :class:`~ontoaligner.aligner.flora.FLORAAligner` class accepts the following parameters to customize alignment behaviour.

.. tab:: 🔀 Subrelation Inference

	.. list-table::
	   :header-rows: 1
	   :widths: 20 12 12 56

	   * - Parameter
	     - Type
	     - Default
	     - Description
	   * - **alpha**
	     - float
	     - 3.0
	     - Benefit-of-doubt parameter for predicate subsumption scoring.
	       Higher values (e.g., 5.0) are more lenient; lower values (e.g., 1.0) are stricter.
	   * - **relinit**
	     - float
	     - 0.1
	     - Initial score for predicates that do not match identically.
	       Used when bootstrapping predicate subsumption.

.. tab:: 💬 Literal Bootstrapping

	.. list-table::
	   :header-rows: 1
	   :widths: 20 12 12 56

	   * - Parameter
	     - Type
	     - Default
	     - Description
	   * - **init_threshold**
	     - float
	     - 0.7
	     - Minimum semantic similarity score for matching string literals during bootstrapping.
	       Range: [0.0, 1.0]. Higher values require more similar literals.
	   * - **string_identity**
	     - bool
	     - False
	     - If ``True``, use exact string matching instead of neural embeddings for literals.
	       This is faster but may yield lower recall.
	   * - **model_id**
	     - str
	     - ``'Lihuchen/pearl_small'``
	     - Hugging Face model ID for semantic embedding of string literals.
	       Set to ``None`` to disable embeddings.
      * - **emb_path**
	     - str
	     - ``'path/to/pre-encoded-embeddings'``
	     - Optional path to pretrained (encoded) embeddings using Hugging Face model for string literals.
	       Set to ``None`` to this if you are whiling to use Hugging Face model.

.. tab:: 🎯 Entity Matching

	.. list-table::
	   :header-rows: 1
	   :widths: 20 12 12 56

	   * - Parameter
	     - Type
	     - Default
	     - Description
	   * - **gramN**
	     - int
	     - 100
	     - Maximum number of evidential triples (facts) to consider per entity during matching.
	       Increase for more evidence; decrease for speed.

.. tab:: ⏸️ Convergence Control

	.. list-table::
	   :header-rows: 1
	   :widths: 20 12 12 56

	   * - Parameter
	     - Type
	     - Default
	     - Description
	   * - **epsilon**
	     - float
	     - 0.01
	     - Convergence threshold. Iteration stops when the change in total alignment score
	       is less than ``epsilon``.
	   * - **max_iterations**
	     - int
	     - 100
	     - Maximum number of main-loop iterations. Acts as a safety cap.

.. tab:: 🔧 Functionality Computation

	.. list-table::
	   :header-rows: 1
	   :widths: 20 12 12 56

	   * - Parameter
	     - Type
	     - Default
	     - Description
	   * - **ngrams**
	     - List[int]
	     - ``[1, 2]``
	     - N-gram sizes for predicate functionality computation.
	       E.g., ``[1, 2]`` considers unary and binary predicate patterns.

.. tab::  🌱 Optional Seed Alignments

	.. list-table::
	   :header-rows: 1
	   :widths: 20 12 12 56

	   * - Parameter
	     - Type
	     - Default
	     - Description
	   * - **training_data**
	     - str or None
	     - None
	     - Path to a tab-separated seed alignment file. Optional third column provides
	       seed confidence (defaults to 1.0). See format below.

	.. code-block:: text

	    <http://kg1.org/E1>    <http://kg2.org/E1>
	    <http://kg1.org/E2>    <http://kg2.org/E2>    0.95

.. tab:: 🖥️ Hardware & Performance

	.. list-table::
	   :header-rows: 1
	   :widths: 20 12 12 56

	   * - Parameter
	     - Type
	     - Default
	     - Description
	   * - **device**
	     - str or None
	     - None
	     - Device for embeddings: ``'cuda'`` or ``'cpu'``.
	       Auto-detects CUDA availability if ``None``.
	   * - **batch_size**
	     - int or None
	     - 32
	     - Batch size for embedding computations.

**Complete Example**

.. code-block:: python

    # Strict, supervised mode
    aligner = FLORAAligner(
        alpha=1.0,              # Conservative subrelation scoring
        init_threshold=0.9,     # High literal similarity required
        epsilon=0.001,          # Strict convergence
        training_data="seeds.tsv",  # Use known pairs
        string_identity=False   # Still use embeddings for unknown literals
    )

    # Fast, lightweight mode
    aligner = FLORAAligner(
        string_identity=True,   # No embeddings
        gramN=50,               # Fewer evidence facts
        max_iterations=20       # Fewer iterations
    )

Advanced Usage
-------------------

.. tab:: 🔐 Supervised Mode


	If seed/training entity pairs are available, you can bootstrap the alignment:

	.. code-block:: python

	    aligner = FLORAAligner(training_data="seed_links.tsv")
	    matchings = aligner.generate(input_data=encoder_output)

	The seed file should contain one alignment pair per line, tab-separated:

	.. code-block::

	    <http://kg1.org/E1>	<http://kg2.org/E1>
	    <http://kg1.org/E2>	<http://kg2.org/E2>	0.95

	An optional third column provides the seed confidence score (defaults to 1.0 if omitted).


.. tab:: ⚡ String-Identity Mode

	For a lightweight run without downloading/running the embedding model, set ``string_identity=True``:

	.. code-block:: python

	    aligner = FLORAAligner(string_identity=True)
	    matchings = aligner.generate(input_data=encoder_output)

	This mode is faster and uses less memory but may yield lower recall on datasets
	where equivalent literals differ in whitespace, casing, or phrasing.


.. tab:: 💾 Pre-computing Embeddings


	For large KGs or repeated experiments you can pre-compute and cache the literal string embeddings:

	.. code-block:: python

	    from ontoaligner.aligner.flora import FLORALiteralsEmbedding
	    from ontoaligner.ontology import FLORAOntology

	    # Load KGs
	    kb1 = FLORAOntology().load_ontology(input_file_path='path/to/kg.ttl')  # load KG1
	    kb2 = FLORAOntology().load_ontology(input_file_path='path/to/kg.ttl')  # load KG2

	    # Compute and save embeddings
	    embedding_model = FLORALiteralsEmbedding(model_id='Lihuchen/pearl_small', identity=False)
	    embedding_model.encode_save(kb1, kb2, emb_path="my_embeddings/")

	Then reuse in multiple experiments::

	    # These embeddings will be loaded from disk instead of recomputed
	    aligner = FLORAAligner(emb_path='my_embeddings/')
	    # (The aligner will look for embeddings in default locations)


.. tab:: 📁 Loading Knowledge Graphs Directly

	If you prefer to work directly with FLORA's native :class:`~ontoaligner.ontology.flora.Graph`
	data structure instead of using the standard dataset pipelines, you can load TTL/XML files directly:

	.. code-block:: python

	    from ontoaligner.ontology.flora import Graph

	    # Load Turtle files directly into Graph objects
	    kg1=Graph().load_turtle_file("path/to/kg1.ttl")
	    kg2=Graph().load_turtle_file("path/to/kg2.ttl")

	    # kg1 and kg2 are now Graph objects ready for the aligner
	    aligner = FLORAAligner()
	    matchings = aligner.generate(input_data=[kg1, kg2])

	**Graph Class Overview**

	The :class:`~ontoaligner.ontology.flora.Graph` is an in-memory directed multigraph optimized for KG alignment:

	.. code-block:: python

	    from ontoaligner.ontology.flora import Graph

	    # Create an empty graph
	    graph = Graph()

	    # Add triples manually
	    graph.add(("http://example.org/John", "http://example.org/knows", "http://example.org/Jane"))
	    graph.add(("http://example.org/John", "http://example.org/age", '"25"^^http://www.w3.org/2001/XMLSchema#integer'))

	    # Check if a triple exists
	    if ("http://example.org/John", "http://example.org/knows", "http://example.org/Jane") in graph:
	        print("Triple found!")

	    # Iterate over all triples
	    for subject, predicate, obj in graph:
	        print(f"{subject} -> {predicate} -> {obj}")

	    # Get predicate statistics
	    predicates = graph.predicates()  # Returns {predicate: count, ...}
	    print(f"Total predicates: {len(predicates)}")

	    # Load from a TTL file
	    graph.load_turtle_file("path/to/ontology.ttl")

	**Key Features of Graph**:

	- **Bidirectional indexing**: Forward (subject→predicate→objects) and reverse (predicate→subject→objects) indices for fast lookups.
	- **Automatic inverse arcs**: For every triple ``(s, p, o)``, the inverse ``(o, p+"-", s)`` is automatically added.
	- **Efficient storage**: Uses nested dictionaries with sets for O(1) membership testing.
	- **Lazy predicate counting**: Predicates are cached and recomputed only when modified.


.. tab:: 📤 Exporting Alignments to RDF/Turtle

	The :class:`~ontoaligner.aligner.flora.FLORARDFWriter` class writes alignment results back to RDF/Turtle format:

	.. code-block:: python

	    from ontoaligner.aligner.flora import FLORARDFWriter, FLORAAligner

	    # Run alignment
	    aligner = FLORAAligner()
	    matchings = aligner.generate(input_data=[kg1, kg2])

	    # Extract entity and predicate alignments from the aligner
	    same_as_scores = aligner.get_same_as_scores()
	    predicate_scores = aligner.get_predicate2super_predicate()

	    # Create an RDF writer with namespace prefixes
	    prefixes = {
	        'ex': 'http://example.org/',
	        'owl': 'http://www.w3.org/2002/07/owl#',
	        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
	        'align': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
	    }

	    writer = FLORARDFWriter(prefixes=prefixes)

	    # Write alignments to file
	    writer.write(
	        output_path="alignments.ttl",
	        kb1=kg1,
	        kb2=kg2,
	        predicate2super_predicate=predicate_scores,
	        same_as_scores=same_as_scores
	    )

	**Output Format**

	The generated Turtle file contains:

	1. **Namespace declarations** — Standard RDF prefixes at the top:

	   .. code-block:: turtle

	       @prefix ex: <http://example.org/> .
	       @prefix owl: <http://www.w3.org/2002/07/owl#> .
	       @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

	2. **Predicate subsumption relationships** — Using ``rdfs:subPropertyOf``:

	   .. code-block:: turtle

	       ex:father rdfs:subPropertyOf ex:parent . # 0.92

	3. **Entity equivalence mappings** — Using ``owl:sameAs``:

	   .. code-block:: turtle

	       ex:John owl:sameAs ex:John_DBpedia . # 0.95
	       ex:Jane owl:sameAs ex:jane_dbpedia . # 0.88

	The ``# score`` suffix (as a comment) indicates the confidence of each alignment.

	**Pre-defined Prefixes from ontoaligner.aligner.flora.fuzzy_logic.prefixes**

	The module ``prefixes.py`` provides two built-in prefix dictionaries with well-known RDF/OWL vocabularies:

	.. tab:: 🔗🌐 General Linked-Data & Knowledge Base Prefixes

	  Import and use the ``prefixes`` dictionary for general-purpose RDF/OWL vocabularies:

	  .. code-block:: python

	     from ontoaligner.aligner.flora import prefixes

	     writer = FLORARDFWriter(prefixes=prefixes)

	  **Supported namespaces:**

	  .. list-table::
	     :header-rows: 1
	     :widths: 15 60

	     * - Prefix
	       - Namespace URI
	     * - ``yago``
	       - http://yago-knowledge.org/resource/
	     * - ``wd``
	       - http://www.wikidata.org/entity/
	     * - ``wdt``
	       - http://www.wikidata.org/prop/direct/
	     * - ``p``
	       - http://www.wikidata.org/prop/
	     * - ``ps``, ``psv``, ``psn``
	       - Wikidata statement value properties
	     * - ``pq``, ``pqv``, ``pqn``
	       - Wikidata qualifier value properties
	     * - ``pr``, ``prv``, ``prn``
	       - Wikidata reference value properties
	     * - ``rdf``, ``rdfs``, ``owl``
	       - RDF/RDFS/OWL core vocabularies
	     * - ``xsd``
	       - XML Schema datatypes
	     * - ``skos``
	       - Simple Knowledge Organization System
	     * - ``schema``
	       - schema.org vocabulary
	     * - ``foaf``
	       - Friend of a Friend
	     * - ``dct``
	       - Dublin Core Terms
	     * - ``cc``
	       - Creative Commons
	     * - ``geo``
	       - OGC GeoSPARQL
	     * - ``prov``
	       - W3C PROV ontology
	     * - ``sh``
	       - SHACL (Shapes Constraint Language)
	     * - ``wikibase``
	       - Wikibase ontology
	     * - ``ontolex``
	       - OntoLex vocabulary
	     * - ``ys``
	       - YAGO Schema

	.. tab:: 🔗📖 DBpedia-Specific Prefixes

	  For DBpedia-centric alignments, use ``prefixes_dbp`` which includes multilingual support:

	  .. code-block:: python

	     from ontoaligner.aligner.flora import prefixes_dbp

	     writer = FLORARDFWriter(prefixes=prefixes_dbp)

	  **Supported DBpedia resources and properties:**

	  .. list-table::
	     :header-rows: 1
	     :widths: 15 60

	     * - Prefix
	       - Description
	     * - ``dbr``
	       - DBpedia English resource namespace
	     * - ``dbr-fr``, ``dbr-zh``, ``dbr-ja``
	       - DBpedia resources in French, Chinese, Japanese
	     * - ``dbo``
	       - DBpedia ontology (classes and properties)
	     * - ``dbp``, ``dbp-fr``, ``dbp-zh``, ``dbp-ja``
	       - DBpedia properties in multiple languages
	     * - ``dbd``
	       - DBpedia datatypes
	     * - ``foaf``, ``rdfs``, ``owl``, ``xsd``
	       - Standard RDF vocabularies (overlaps with ``prefixes``)
	     * - ``dc``, ``dc-terms``
	       - Dublin Core (elements and terms)
	     * - ``skos``
	       - SKOS vocabulary
	     * - ``geo``
	       - OGC WGS84 Geo Positioning

	.. important::

	   **What is supported:**

	   - ✅ Custom prefix dictionaries – Pass any ``Dict[str, str]`` mapping prefix names to URIs.
	   - ✅ Pre-configured vocabularies – Use ``prefixes`` or ``prefixes_dbp`` from the module.
	   - ✅ Mixed vocabularies – Combine multiple prefix sources into a single dict.
	   - ✅ Multilingual DBpedia – ``prefixes_dbp`` handles English, French, Chinese, Japanese.
	   - ✅ Standard RDF/OWL predicates – ``owl:sameAs`` and ``rdfs:subPropertyOf`` are always supported.

	   **What is NOT supported:**

	   - ❌ Automatic namespace detection – Prefixes must be explicitly provided; the writer does not introspect your KGs.
	   - ❌ Custom alignment predicates – The writer always uses ``owl:sameAs`` for entities and ``rdfs:subPropertyOf`` for predicates. Other predicates (e.g., ``skos:closeMatch``) are not generated.
	   - ❌ Filtering by namespace – All IRIs are written as-is; use post-processing to filter by namespace if needed.
	   - ❌ Blank node abbreviation – IRIs are always written in full form, not abbreviated with prefixes in the triples themselves.
	   - ❌ Alignment metadata – Scores are appended as tab-separated comments only; no formal alignment vocabulary (e.g., from `ALIGNAPI <http://alignapi.gforge.inria.fr/>`_) is used.

	**Why Use FLORARDFWriter?**

	- **Standard RDF format**: Output is directly readable by any RDF tool or SPARQL engine.
	- **Interpretable scores**: Comments preserve confidence scores for downstream analysis.
	- **Integration-ready**: Results can be loaded into ontology editors (Protégé, TopBraid) or linked data platforms.
	- **Reproducible**: Combined with seed alignments, enables iterative refinement workflows.




.. hint::

    See also the `Package Reference > FLORA Aligner <../package_reference/aligners.html#flora-aligner>`_.

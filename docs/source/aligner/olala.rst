OLaLa: OM with LLMs
=====================================================

OLaLa
--------

**OLaLa** (\ **O**\ ntology matching with **La**\ rge **La**\ nguage models) is a retrieval-augmented ontology alignment system
that combines dense semantic retrieval with open-source decoder language models to verify candidate correspondences.
Key properties of OLaLa are:

1. *Zero-shot / few-shot* — Requires no labelled training pairs; a small set of in-context examples is sufficient.
2. *Open-source LLMs only* — All results are reproducible; no paid API is involved.
3. *Confidence-calibrated* — Binary yes/no token probabilities are normalised into a ``[0, 1]`` confidence score.
4. *High-precision safety net* — An exact-match high-precision matcher supplements the LLM to recover trivial correspondences at full confidence.
5. *Postprocessing pipeline* — Bad-host filtering, maximum-weight bipartite extraction, and confidence thresholding produce a clean one-to-one alignment.

The following diagram (Figure 1 of the paper) illustrates the overall OLaLa pipeline:

.. raw:: html

    <div align="center">
      <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/refs/heads/dev/docs/source/img/olala.png" width="80%"/>
    </div>


Given two ontologies :math:`O_1` and :math:`O_2`, OLaLa produces a set of correspondence pairs :math:`M = \{(c, c', s) \mid c \in O_1,\; c' \in O_2,\; s \in [0,1]\}`, where :math:`s` is the LLM-derived confidence that concepts :math:`c` and :math:`c'` refer to the same real-world entity. The pipeline has four stages:

**🔍 1. Candidate Generation (SBERT)**: Each ontology concept is verbalized into one or more text strings using the **TextExtractorSet** strategy — extracting labels, descriptions, annotation-property texts, and the URI fragment (when it contains fewer than 50 % digits). All texts per resource are embedded with a Sentence-BERT model and a bidirectional cosine-similarity search returns the top-*k* candidates per resource. The default model is ``multi-qa-mpnet-base-dot-v1``, and *k* = 5. The procedure is run in both directions (``source → target and target → source``), and the union of candidates is kept.

**🤖 2. LLM Binary Verification**: Each candidate pair is presented to a decoder LLM via a **few-shot prompt** (``prompt 7`` in the paper — see the *Prompts* section below):

.. code-block:: text

	Classify if two descriptions refer to the same real world entity (ontology matching).
	### Concept one: endocrine pancreas secretion    ### Concept two: Pancreatic Endocrine Secretion ### Answer: yes
	### Concept one: urinary bladder urothelium      ### Concept two: Transitional Epithelium         ### Answer: no
	### Concept one: trigeminal V nerve ophthalmic division ### Concept two: Ophthalmic Nerve         ### Answer: yes
	### Concept one: foot digit 1 phalanx            ### Concept two: Foot Digit 2 Phalanx           ### Answer: no
	### Concept one: large intestine                 ### Concept two: Colon                          ### Answer: no
	### Concept one: ocular refractive media         ### Concept two: Refractile Media               ### Answer: yes
	### Concept one: {left}                          ### Concept two: {right}                        ### Answer:

Generation stops as soon as a ``yes`` / ``no`` (or ``true`` / ``false``) token is produced. The softmax probability of the positive class is normalised by the sum of positive and negative class probabilities to yield a confidence score: :math:`s = p_{yes}/(p_{yes} + p_{no})`, where every correspondence with :math:`s \geq 0.5` is treated as a positive match.

**🎯 3. High-Precision Matching**: In parallel, an exact-match **high-precision matcher** independently finds concepts with identical normalized labels or URI fragments (lowercased, camel-case split, non-alphanumeric characters removed). Only unambiguous 1:1 pairs (no N:M conflicts) are kept, all at confidence 1.0. These are merged into the LLM output to ensure trivial correspondences are never missed.

**🧹 4. Postprocessing:** The merged alignment is cleaned in three steps:

- **Bad-host filter** — removes correspondences whose IRIs do not belong to the
  expected source or target ontology hosts.
- **Maximum-weight bipartite extraction** — enforces a one-to-one mapping by
  solving the assignment problem with ``scipy.optimize.linear_sum_assignment``.
- **Confidence filter** — discards all correspondences below a configurable threshold
  (default 0.5).

.. note::

    **Reference:** Sven Hertling and Heiko Paulheim. 2023. OLaLa: Ontology Matching with Large Language Models. In Proceedings of the 12th Knowledge Capture Conference 2023 (K-CAP '23). Association for Computing Machinery, New York, NY, USA, 131–139. https://doi.org/10.1145/3587259.3627571


Usage
-----

.. tab:: ➡️ 1: Import

    Import the OLaLa pipeline components and utility modules.

    .. code-block:: python

        import json
        from ontoaligner.ontology import OLaLaOMDataset
        from ontoaligner.encoder import OLaLaEncoder
        from ontoaligner.aligner.olala import (
            OLaLaSBERTRetrieval,
            OLaLaLLMAligner,
            OLaLaHighPrecisionMatcher,
            OLaLaAligner,
        )
        from ontoaligner.aligner.olala.postprocessor import olala_postprocessor
        from ontoaligner.utils import metrics, xmlify

    .. note::

        ``OLaLaAligner`` is a thin orchestrator that wires together the retriever,
        the LLM aligner, and the high-precision matcher.
        Each component can also be used independently.

.. tab:: ➡️ 2: Parse Ontologies

    ``OLaLaOMDataset.collect()`` calls ``OLaLaOntology.parse()`` for each OWL file.
    The parser extracts standard fields *and* OLaLa-specific fields (TextExtractorSet,
    OnlyLabel, high-precision texts, host) into an ``"olala"`` sub-dictionary
    on every concept.

    .. code-block:: python

        task = OLaLaOMDataset(language="en")
        print("Task:", task)

        dataset = task.collect(
            source_ontology_path="assets/source.owl",
            target_ontology_path="assets/target.owl",
            reference_matching_path="assets/reference.xml",  # optional, for evaluation
        )

    Each entry in ``dataset["source"]`` / ``dataset["target"]`` has the shape:

    .. code-block::

        {
            "iri":     "http://purl.obolibrary.org/obo/MA_0000001",
            "label":   "mouse",
            "olala": {
                "text_extractor_set":            ["mouse", ...],
                "normalized_text_extractor_set": ["mouse", ...],
                "only_label":                    "mouse",
                "hp_texts":                      ["mouse"],
                "host":                          "purl.obolibrary.org",
                "normalized_label":              "mouse",
                "normalized_uri_fragment":       "ma 0000001",
                ...
            }
        }

    .. warning::

        Only OWL/XML ontologies are supported out of the box.
        For other RDF serializations, supply a custom parser.

.. tab:: ➡️ 3: Encode Ontologies

    ``OLaLaEncoder`` converts the parsed ontology items into the flat lists expected
    by the retriever, LLM aligner, and high-precision matcher.

    .. code-block:: python

        encoder_model = OLaLaEncoder()

        encoded_ontology = encoder_model(
            source=dataset["source"],
            target=dataset["target"],
        )
        # encoded_ontology == [source_items, target_items]

    Each item in ``source_items`` / ``target_items`` exposes the fields
    ``texts``, ``only_label``, ``hp_texts``, ``keep_for_sbert``, and ``expected_host``
    used by the downstream components.

.. tab:: ➡️ 4: Initialise Components

    Instantiate the three OLaLa components — retriever, LLM aligner, and
    high-precision matcher — then wire them into ``OLaLaAligner``.

    .. code-block:: python

        # SBERT candidate retriever
        retriever = OLaLaSBERTRetrieval(
            device="cuda",
            top_k=5,
            both_directions=True,
            topk_per_resource=True,
        )

        # LLM binary verifier
        llm_aligner = OLaLaLLMAligner(
            device="cuda",
            max_new_tokens=10,
            temperature=0.0,
            truncation=True,
            max_length=2048,
            padding=True,
            loading_arguments={
                "device_map": "auto",
                "torch_dtype": "torch.float16",
            },
        )

        # High-precision exact matcher
        hp_aligner = OLaLaHighPrecisionMatcher(confidence=1.0)

        # Orchestrator
        olala = OLaLaAligner(
            retriever=retriever,
            llm_aligner=llm_aligner,
            hp_aligner=hp_aligner,
        )

    See `Configuration`_ below for a complete parameter reference.

.. tab:: ➡️ 5: Load Models and Generate Alignments

    Load the SBERT and LLM weights, then call ``generate()``.

    .. code-block:: python

        olala.load(
            llm_path="upstage/Llama-2-70b-instruct-v2",
            retriever_path="multi-qa-mpnet-base-dot-v1",
        )

        alignments = olala.generate(input_data=encoded_ontology)

    The raw output is a flat list of grouped LLM predictions and high-precision
    correspondences, each tagged with an ``alignment_type`` field:

    .. code-block::

        [
            {
                "alignment_type": "rag",
                "source": "http://example.org/A",
                "target-cands": ["http://example.org/B", ...],
                "score-cands":  [0.87, ...]
            },
            {
                "alignment_type": "hp",
                "source": "http://example.org/C",
                "target": "http://example.org/D",
                "score":  1.0
            },
            ...
        ]

.. tab:: ➡️ 6: Postprocess

    ``olala_postprocessor`` merges LLM and high-precision predictions, applies
    host filtering, extracts a one-to-one alignment, and applies the confidence threshold.

    .. code-block:: python

        final_matchings = olala_postprocessor(
            alignments,
            encoded_ontology,
            confidence_threshold=0.5,
            strict_bad_hosts=False,
        )

    The output is a clean list of flat correspondences:

    .. code-block::

        [
            {"source": "http://example.org/A", "target": "http://example.org/B", "score": 0.87},
            ...
        ]

.. tab:: ➡️ 7: Evaluate and Export

    Compare predictions to a reference alignment and export results.

    .. code-block:: python

        # Evaluate
        evaluation = metrics.evaluation_report(
            predicts=final_matchings,
            references=dataset["reference"],
        )
        print("OLaLa Evaluation Report:")
        print(json.dumps(evaluation, indent=4))

    Example output:

    .. code-block::

        {
            "intersection": 1317,
            "precision":     89.4,
            "recall":        89.1,
            "f-score":       90.2,
            "predictions-len": 1478,
            "reference-len": 1478
        }

    Export the final alignment to XML (OAEI-compatible) or JSON:

    .. tab:: 📄 Export to XML

        .. code-block:: python

            xml_str = xmlify.xml_alignment_generator(matchings=final_matchings)
            with open("olala_matchings.xml", "w", encoding="utf-8") as f:
                f.write(xml_str)

    .. tab:: 🧾 Export to JSON

        .. code-block:: python

            with open("olala_matchings.json", "w", encoding="utf-8") as f:
                json.dump(final_matchings, f, indent=4, ensure_ascii=False)


Configuration
-------------

.. tab:: 🔍 OLaLaSBERTRetrieval

    .. list-table::
       :header-rows: 1
       :widths: 22 12 14 52

       * - Parameter
         - Type
         - Default
         - Description
       * - **device**
         - str
         - ``"cpu"``
         - Device for the SentenceTransformers model (``"cpu"`` or ``"cuda"``).
       * - **top_k**
         - int
         - ``5``
         - Number of candidate targets retrieved per source resource.
           Higher values increase recall but increase LLM inference cost.
       * - **both_directions**
         - bool
         - ``True``
         - If ``True``, retrieval is run in both source→target and target→source
           directions and the union is taken.
       * - **topk_per_resource**
         - bool
         - ``True``
         - If ``True``, top-*k* filtering is applied per resource after merging
           both directions, preventing any single resource from dominating.

.. tab:: 🤖 OLaLaLLMAligner

    .. list-table::
       :header-rows: 1
       :widths: 22 12 14 52

       * - Parameter
         - Type
         - Default
         - Description
       * - **device**
         - str
         - ``"cpu"``
         - Device for the language model (``"cpu"`` or ``"cuda"``).
       * - **max_new_tokens**
         - int
         - ``10``
         - Maximum number of tokens the model is allowed to generate per prompt.
           Generation usually stops early when a yes/no token is detected.
       * - **temperature**
         - float
         - ``0.0``
         - Sampling temperature. Set to ``0.0`` for fully greedy (deterministic) decoding.
       * - **word_stopper**
         - bool
         - ``True``
         - If ``True``, generation stops immediately after the first yes/no token.
           Disable only for debugging or custom stopping strategies.
       * - **loading_arguments**
         - dict
         - ``{}``
         - Extra keyword arguments forwarded to ``AutoModelForCausalLM.from_pretrained``.
           Common keys: ``device_map``, ``torch_dtype``, ``load_in_8bit``.
       * - **system_prompt_template**
         - str
         - ``"{user_prompt}"``
         - Optional wrapper around the filled prompt.
           Use this to add a system message for chat-tuned models,
           e.g. ``"[INST] {user_prompt} [/INST]"``.
       * - **dataset_class**
         - type
         - ``OLaLaLLMDataset``
         - Dataset class used to build prompts. Override to customise text verbalization.
       * - **truncation**
         - bool
         - ``True``
         - Whether to truncate inputs that exceed ``max_length``.
       * - **max_length**
         - int
         - ``2048``
         - Maximum tokenized input length.
       * - **padding**
         - bool
         - ``True``
         - Whether to pad inputs to the same length within a batch.

.. tab:: 🎯 OLaLaHighPrecisionMatcher

    .. list-table::
       :header-rows: 1
       :widths: 22 12 14 52

       * - Parameter
         - Type
         - Default
         - Description
       * - **confidence**
         - float
         - ``1.0``
         - Confidence score assigned to every exact correspondence produced by
           this matcher. Should remain at ``1.0`` in most use cases.

.. tab:: 🧹 olala_postprocessor

    .. list-table::
       :header-rows: 1
       :widths: 25 12 12 51

       * - Parameter
         - Type
         - Default
         - Description
       * - **alignments**
         - list
         - —
         - Raw output of ``OLaLaAligner.generate()``.
       * - **encoded_ontology**
         - list
         - —
         - ``[source_items, target_items]`` from ``OLaLaEncoder``.
           Used to derive expected ontology hosts for bad-host filtering.
       * - **confidence_threshold**
         - float
         - ``0.5``
         - Correspondences with scores below this value are discarded.
           The default removes all pairs where the LLM preferred ``no``.
       * - **strict_bad_hosts**
         - bool
         - ``False``
         - If ``True``, correspondences whose source or target IRI host cannot be
           determined are also removed.
           Set to ``True`` when the ontologies have stable, well-known hosts.

**Complete Configuration Example**

.. code-block:: python

    retriever = OLaLaSBERTRetrieval(
        device="cuda",
        top_k=5,
        both_directions=True,
        topk_per_resource=True,
    )

    llm_aligner = OLaLaLLMAligner(
        device="cuda",
        max_new_tokens=10,
        temperature=0.0,
        truncation=True,
        max_length=2048,
        padding=True,
        system_prompt_template="[INST] {user_prompt} [/INST]",
        loading_arguments={
            "device_map": "auto",
            "torch_dtype": "torch.float16",
            "load_in_8bit": True,
        },
    )

    hp_aligner = OLaLaHighPrecisionMatcher(confidence=1.0)

    olala = OLaLaAligner(
        retriever=retriever,
        llm_aligner=llm_aligner,
        hp_aligner=hp_aligner,
    )


Prompts
-------

OLaLa supports both zero-shot and few-shot prompting strategies. The table below summarises
the prompts evaluated in the paper's ablation study on the anatomy track.
Prompt 7 (the default few-shot prompt) achieves the best balance between F-measure and runtime.

.. list-table::
   :header-rows: 1
   :widths: 6 55 9 9 9 12

   * - ID
     - Prompt template
     - Prec
     - Rec
     - F1
     - Time
   * - 0 *(zero-shot)*
     - ``Classify if the following two concepts are the same.\n### First concept:\n{left}\n### Second concept:\n{right}\n### Answer:``
     - 0.853
     - 0.866
     - 0.861
     - 4h 19m
   * - 7 *(default)*
     - Few-shot with 3 positive + 3 negative examples and task description (see below)
     - 0.914
     - 0.891
     - 0.902
     - 2h 41m

The default **prompt 7** used by ``OLaLaLLMDataset`` is:

.. code-block:: text

    Classify if two descriptions refer to the same real world entity (ontology matching).
    ### Concept one: endocrine pancreas secretion    ### Concept two: Pancreatic Endocrine Secretion ### Answer: yes
    ### Concept one: urinary bladder urothelium      ### Concept two: Transitional Epithelium         ### Answer: no
    ### Concept one: trigeminal V nerve ophthalmic division ### Concept two: Ophthalmic Nerve         ### Answer: yes
    ### Concept one: foot digit 1 phalanx            ### Concept two: Foot Digit 2 Phalanx           ### Answer: no
    ### Concept one: large intestine                 ### Concept two: Colon                          ### Answer: no
    ### Concept one: ocular refractive media         ### Concept two: Refractile Media               ### Answer: yes
    ### Concept one: {left}                          ### Concept two: {right}                        ### Answer:

.. note::

    To use a custom prompt, subclass ``OLaLaLLMDataset``, override the ``prompt`` class attribute,
    and pass your subclass via the ``dataset_class`` argument of ``OLaLaLLMAligner``.


Advanced Usage
--------------

.. sidebar:: ``{user_prompt}`` placeholder

    The ``{user_prompt}`` placeholder is replaced at runtime by the filled
    few-shot prompt (including both the examples and the ``{left}``/``{right}``
    pair under evaluation).

**🔧 Custom System Prompt (Chat Models) Usage**: Chat-tuned models such as ``Llama-2-70b-chat-hf`` expect a specific conversation template. Pass ``system_prompt_template`` to wrap the filled few-shot prompt:

.. code-block:: python

    llm_aligner = OLaLaLLMAligner(
        system_prompt_template="[INST] {user_prompt} [/INST]",
        ...
    )

**⚡ Lightweight / CPU Mode Usage**: For quick experiments without GPU access, reduce the model size and disable 8-bit loading:

.. code-block:: python

    retriever = OLaLaSBERTRetrieval(device="cpu", top_k=3)

    llm_aligner = OLaLaLLMAligner(
        device="cpu",
        loading_arguments={"torch_dtype": "torch.float32"},
    )

Consider using a smaller model such as ``meta-llama/Llama-2-7b-hf`.

**🔬 Components Standalone Usage**: Each component can be used independently of ``OLaLaAligner``:

.. sidebar:: OLaLa High Precision Matcher

    Because the high-precision matcher only produces correspondences for concepts that share
    an *identical normalized label or URI fragment*, its **precision is typically 1.0** but
    its **recall is limited** to concepts with lexically identical surface forms.
    For broader coverage, use it as a complement to the full OLaLa pipeline
    (which is the default behaviour of ``OLaLaAligner``).


.. code-block:: python

    # SBERT retrieval only
    retriever = OLaLaSBERTRetrieval(device="cuda", top_k=5)
    retriever.load(path="multi-qa-mpnet-base-dot-v1")
    candidates = retriever.generate(input_data=encoded_ontology)

    # LLM aligner only (accepts SBERT candidates)
    llm_aligner = OLaLaLLMAligner(device="cuda", ...)
    llm_aligner.load(path="upstage/Llama-2-70b-instruct-v2")
    llm_predictions = llm_aligner.generate(
        input_data=[source_items, target_items, candidates]
    )

    # High-precision matcher only
    hp_aligner = OLaLaHighPrecisionMatcher(confidence=1.0)
    hp_predictions = hp_aligner.generate(input_data=encoded_ontology)

**When to use the standalone matcher**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Scenario
     - Recommendation
   * - Fast baseline / sanity check
     - Run standalone; takes seconds even on large ontologies.
   * - Ontologies with highly consistent labelling conventions
     - Standalone HighPrecisionMatcher may already achieve acceptable recall.
   * - Pre-filtering before a costly LLM run
     - Run HP first, remove matched concepts, then feed the remainder to ``OLaLaAligner``.
   * - Full production alignment
     - Use ``OLaLaAligner`` — HP is automatically included and its results are merged
       with LLM predictions at confidence 1.0.


.. hint::

    See also the `Package Reference > OLaLa Aligner <../package_reference/aligners.html#olala-aligner>`_.

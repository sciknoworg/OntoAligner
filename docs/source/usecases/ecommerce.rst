Product Alignment in eCommerce
==============================

This tutorial explains how to align product categories between two e‑commerce ontologies (Amazon vs. eBay)
using a hybrid pipeline of fuzzy matching, SBERT‑style retrieval, and LLM‑based decoding. We illustrate
each step with sample RDF/XML datasets, show result tables, and discuss the objective of achieving
semantic interoperability for product taxonomies.

Objective
---------

In modern e‑commerce, different platforms use heterogeneous product taxonomies that hinder unified search,
recommendation, and analytics. An ontology alignment pipeline discovers correspondences between source
and target classes—e.g. mapping `GamingLaptop` (Amazon) to `GamingNotebook` (eBay)—to enable cross‑site
product integration and comparison.

Sample Datasets
---------------

Below are excerpts from the Amazon and eBay ontologies. Each defines a root category, sub‑classes, and properties.

.. list-table::
   :header-rows: 1

   * - Ontology   - Root class    - Sub‑classes                                        - Properties
   * - Amazon     - Electronics    - Laptop, GamingLaptop, Ultrabook, Smartphone         - hasBrand, hasManufacturer, screenSize
   * - eBay       - Computers      - Notebook, GamingNotebook, BusinessNotebook          - manufacturer, producer, displaySize

Amazon ontology (simplified):

.. code-block:: xml

    <owl:Class rdf:about="...#Laptop">
      <skos:prefLabel>Laptop</skos:prefLabel>
      <skos:broader rdf:resource="...#Electronics"/>
    </owl:Class>

eBay ontology (simplified):

.. code-block:: xml

    <owl:Class rdf:about="...#Notebook">
      <skos:prefLabel>Notebook</skos:prefLabel>
      <skos:broader rdf:resource="...#Computers"/>
    </owl:Class>

Step 1: Load and Parse Ontologies
----------------------------------

We begin by parsing the RDF/XML representations of the Amazon and eBay ontologies using the `GenericOntology` class. This process extracts classes, labels, hierarchical relationships, synonyms, and comments, structuring them into a format suitable for alignment tasks.

.. code-block:: python

    from ontoaligner.ontology import GenericOntology
    ontology = GenericOntology()
    src_onto = ontology.parse("amazon.owl")
    tgt_onto = ontology.parse("ebay.owl")

**Sample Parsed Output:**

.. code-block:: json

    [
      {
        "name": "Electronics",
        "iri": "http://example.org/amazon#Electronics",
        "label": "Electronics",
        "childrens": [
          {
            "iri": "http://example.org/amazon#Laptop",
            "label": "Laptop",
            "name": "Laptop"
          }
        ],
        "parents": [],
        "synonyms": [],
        "comment": []
      },
      {
        "name": "Laptop",
        "iri": "http://example.org/amazon#Laptop",
        "label": "Laptop",
        "childrens": [
          {
            "iri": "http://example.org/amazon#GamingLaptop",
            "label": "GamingLaptop",
            "name": "GamingLaptop"
          },
          {
            "iri": "http://example.org/amazon#Ultrabook",
            "label": "Ultrabook",
            "name": "Ultrabook"
          }
        ],
        "parents": [],
        "synonyms": [],
        "comment": []
      }
    ]

Step 2: Encode Concepts with Lightweight Encoder
------------------------------------------------

To facilitate efficient matching, we encode each concept by concatenating its label with its parent labels. This approach captures both the concept's identity and its hierarchical context, providing a richer representation for similarity computations.

.. code-block:: python

    from ontoaligner.encoder import ConceptParentLightweightEncoder
    encoder = ConceptParentLightweightEncoder()
    encoder_output = encoder(source=src_onto, target=tgt_onto)

**Sample Encoder Output:**

.. code-block:: json

    [
      [
        {"iri": "http://example.org/amazon#Electronics", "text": "electronics"},
        {"iri": "http://example.org/amazon#Laptop", "text": "laptop electronics"},
        {"iri": "http://example.org/amazon#GamingLaptop", "text": "gaminglaptop laptop electronics"},
        {"iri": "http://example.org/amazon#Ultrabook", "text": "ultrabook laptop electronics"},
        {"iri": "http://example.org/amazon#Smartphone", "text": "smartphone electronics"}
      ],
      [
        {"iri": "http://example.org/ebay#Computers", "text": "computers"},
        {"iri": "http://example.org/ebay#Notebook", "text": "notebook computers"},
        {"iri": "http://example.org/ebay#GamingNotebook", "text": "gamingnotebook notebook computers"},
        {"iri": "http://example.org/ebay#BusinessNotebook", "text": "businessnotebook notebook computers"}
      ]
    ]

Step 3: Simple Fuzzy String Matching
-------------------------------------

We apply a fuzzy string matching algorithm to identify potential correspondences based on lexical similarity. This method computes similarity scores between concept labels, capturing straightforward matches.

.. code-block:: python

    from ontoaligner.aligner import SimpleFuzzySMLightweight
    fuzzy = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.4)
    fuzzy_matches = fuzzy.generate(input_data=encoder_output)

**Fuzzy Matchings:**

.. list-table::
   :header-rows: 1

   * - Source               - Target             - Score
   * - Electronics          - Computers          - 0.40
   * - GamingLaptop         - GamingNotebook     - 0.55
   * - Ultrabook            - BusinessNotebook   - 0.47

Step 4: SBERT‑Style Retrieval Matching
--------------------------------------

To capture semantic similarities beyond lexical matching, we utilize a Sentence-BERT (SBERT) model. SBERT encodes concepts into dense vector representations, allowing for semantic similarity computations.

.. code-block:: python

    from ontoaligner.aligner import SBERTRetrieval
    sbert = SBERTRetrieval(device="cpu", top_k=3)
    sbert.load(path="all-MiniLM-L6-v2")
    sbert_matches = sbert.generate(input_data=encoder_output)

**SBERT Top‑3 Matches (examples):**

.. list-table::
   :header-rows: 1

   * - Source        - Top Matches (target, score)
   * - Electronics   - Computers (0.66), Notebook (0.41), GamingNotebook (0.13)
   * - Laptop        - Notebook (0.73), Computers (0.62), GamingNotebook (0.47)
   * - GamingLaptop  - GamingNotebook (0.63), Notebook (0.46), Computers (0.29)

Step 5: Post‑process Retrieval Outputs
--------------------------------------

We refine the SBERT matching results by filtering and selecting the most relevant correspondences based on similarity scores and domain knowledge.

.. code-block:: python

    from ontoaligner.postprocess import retriever_postprocessor
    sbert_clean = retriever_postprocessor(sbert_matches)

**Cleaned SBERT matches:**

.. list-table::
   :header-rows: 1

   * - Source        - Target           - Score
   * - Electronics   - Computers        - 0.66
   * - Electronics   - Notebook         - 0.41
   * - Laptop        - Notebook         - 0.73
   * - GamingLaptop  - GamingNotebook   - 0.63
   * - Ultrabook     - Notebook         - 0.61

Step 6: LLM‑Based Decoding for Complex Cases
---------------------------------------------

For complex or ambiguous cases where previous methods may fall short, we employ a Large Language Model (LLM) to generate potential alignments. The LLM considers broader context and domain knowledge to suggest matches.

.. code-block:: python

    from ontoaligner.encoder import ConceptLLMEncoder
    from ontoaligner.aligner import AutoModelDecoderLLM, ConceptLLMDataset

    llm_enc = ConceptLLMEncoder()
    src_ctx, tgt_ctx = llm_enc(source=src_onto, target=tgt_onto)

    ds = ConceptLLMDataset(source_onto=src_ctx, target_onto=tgt_ctx)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=512, collate_fn=ds.collate_fn)

    llm = AutoModelDecoderLLM(device="cuda", max_length=200, max_new_tokens=5)
    llm.load(path="Qwen/Qwen2-0.5B")

    preds = []
    from tqdm import tqdm
    for batch in tqdm(dl):
        seqs = llm.generate(batch["prompts"])
        preds.extend(seqs)

Step 7: Post‑process LLM Predictions
-------------------------------------

We process the LLM-generated predictions using a TF-IDF-based label mapper and a logistic regression classifier to determine the most probable alignments.

.. code-block:: python

    from ontoaligner.postprocess import TFIDFLabelMapper, llm_postprocessor
    from sklearn.linear_model import LogisticRegression

    mapper = TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1,1))
    llm_matches = llm_postprocessor(predicts=preds, mapper=mapper, dataset=ds)

**Sample LLM Matches:**

.. list-table::
   :header-rows: 1

   * - Source        - Target           - Confidence
   * - Smartphone    - MobilePhone      - 0.85
   * - Ultrabook     - BusinessNotebook - 0.78

Step 8: Consolidate and Visualize Results
-----------------------------------------

We aggregate the matches from all methods to provide a comprehensive view of the alignments and their respective confidence scores.

.. list-table::
   :header-rows: 1

   * - Method      - Pairs Matched  - Example Pair                          - Avg. Score
   * - Fuzzy       - 3              - GamingLaptop ↔ GamingNotebook         - 0.47
   * - SBERT       - 5              - Laptop ↔ Notebook                     - 0.63
   * - LLM

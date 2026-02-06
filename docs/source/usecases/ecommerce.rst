eCommerce
=================
.. hint::

        **We are using small sample ontologies within this tutorial.**

.. sidebar:: Colab Notebook

    .. raw:: html

        e-Commerce Tutorial On:
        <a href="https://colab.research.google.com/drive/1FgKL-D6IySlsDU58XVfnRjH9dFsykgoT">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="e-Commerce Tutorial">
        </a>



This tutorial explains how to align product categories between two e‑commerce ontologies (Amazon vs. eBay) using a hybrid pipeline of fuzzy matching, SBERT‑style retrieval, and LLM‑based decoding. We illustrate each step with sample RDF/XML datasets, show result tables, and discuss the objective of achieving semantic interoperability for product taxonomies.



**Objective**: In modern e‑commerce, different platforms use heterogeneous product taxonomies that hinder unified search, recommendation, and analytics. An ontology alignment pipeline discovers correspondences between source and target classes—e.g. mapping ``GamingLaptop`` (from Amazon) to ``GamingNotebook`` (to eBay)—to enable cross‑site product integration and comparison.

Lets see samples (according to the following table) from Amazon and eBay ontologies that are suitable for our objective. Each cass as a root category for the product can be divided into multiple sub-categories and they have a different properties.




.. list-table:: Sample from the **Amazon** and **eBay** Ontologies.
   :header-rows: 1

   * - Ontology
     - Category
     - Sub‑Category
     - Properties
   * - `Amazon Ontology <https://github.com/sciknoworg/OntoAligner/tree/main/assets/e-commerce/amazon.owl>`_
     - ``Electronics``
     - ``Laptop``, ``GamingLaptop``, ``Ultrabook``, ``Smartphone``
     - ``hasBrand``, ``hasManufacturer``, ``screenSize``
   * - `eBay Ontology <https://github.com/sciknoworg/OntoAligner/tree/main/assets/e-commerce/ebay.owl>`_
     - ``Computers``
     - ``Notebook``, ``GamingNotebook``, ``BusinessNotebook``
     - ``manufacturer``, ``producer``, ``displaySize``


The visualization below illustrates how ontologies and their corresponding alignment are represented.

.. raw:: html

   <iframe src="../_static/amazon-ebay-alignment.html.html"
           width="90%"
           height="900"
           style="border:none;">
   </iframe>


.. note::

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


Parsing Ontology
----------------------------------

We begin by parsing the RDF/XML representations of the Amazon and eBay ontologies using the ``GenericOntology`` class. This process extracts classes, labels, hierarchical relationships, synonyms, and comments, structuring them into a format suitable for alignment tasks.

.. code-block:: python

    from ontoaligner.ontology import GenericOntology

    ontology = GenericOntology()

    src_onto = ontology.parse("amazon.owl")
    tgt_onto = ontology.parse("ebay.owl")

Sample Parsed Output:

.. code-block:: javascript

    [
      {
        "name": "Electronics",
        "iri": "http://example.org/amazon#Electronics",
        "label": "Electronics",
        "childrens": [{
            "iri": "http://example.org/amazon#Laptop",
            "label": "Laptop",
            "name": "Laptop"
        }],
        "parents": [],
        "synonyms": [],
        "comment": []
      },
      ...
    ]

Apply Encoder
------------------------------------------------

To facilitate efficient matching, we encode each concept by concatenating its label with its parent labels. This approach captures both the concept's identity and its hierarchical context, providing a richer representation for similarity computations.

.. code-block:: python

    from ontoaligner.encoder import ConceptParentLightweightEncoder

    encoder = ConceptParentLightweightEncoder()

    encoder_output = encoder(source=src_onto, target=tgt_onto)

Sample Encoder Output:

.. code-block:: javascript

    [
      [
        {"iri": "http://example.org/amazon#Electronics", "text": "electronics"},
        {"iri": "http://example.org/amazon#Laptop", "text": "laptop electronics"},
        {"iri": "http://example.org/amazon#GamingLaptop", "text": "gaminglaptop laptop electronics"},
        {"iri": "http://example.org/amazon#Ultrabook", "text": "ultrabook laptop electronics"},
        {"iri": "http://example.org/amazon#Smartphone", "text": "smartphone electronics"}
      ],
      ...
    ]

Lightweight Aligner
-------------------------------------

We apply a fuzzy string matching algorithm to identify potential correspondences based on lexical similarity. This method computes similarity scores between concept labels, capturing straightforward matches.

.. code-block:: python

    from ontoaligner.aligner import SimpleFuzzySMLightweight

    fuzzy = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.4)

    fuzzy_matches = fuzzy.generate(input_data=encoder_output)

.. list-table:: Lightweight Aligner - Example Matches
   :header-rows: 1

   * - Source Ontology (Amazon)
     - Target Ontology (eBay)
     - Similarity Score
   * - Electronics
     - Computers
     - 0.40
   * - GamingLaptop
     - GamingNotebook
     - 0.55
   * - Ultrabook
     - BusinessNotebook
     - 0.47

Retrieval Aligner
--------------------------------------

To capture semantic similarities beyond lexical matching, we utilize a Sentence-BERT (SBERT) model. SBERT encodes concepts into dense vector representations, allowing for semantic similarity computations.

.. code-block:: python

    from ontoaligner.aligner import SBERTRetrieval

    sbert = SBERTRetrieval(device="cpu", top_k=3)

    sbert.load(path="all-MiniLM-L6-v2")

    sbert_matches = sbert.generate(input_data=encoder_output)

.. list-table:: Retrieval Aligner - SBERT Top‑3 Matches (examples)
   :header-rows: 1

   * - Source
     - Top Matches (target category, similarity score)
   * - Electronics
     - Computers (0.66), Notebook (0.41), GamingNotebook (0.13)
   * - Laptop
     - Notebook (0.73), Computers (0.62), GamingNotebook (0.47)
   * - GamingLaptop
     - GamingNotebook (0.63), Notebook (0.46), Computers (0.29)

**Retrieval Post‑processing:** We refine the SBERT matching results by filtering and selecting the most relevant correspondences based on similarity scores and domain knowledge.

.. code-block:: python

    from ontoaligner.postprocess import retriever_postprocessor

    sbert_clean = retriever_postprocessor(sbert_matches)


.. list-table:: Retrieval Aligner - Cleaned SBERT matches
   :header-rows: 1

   * - Source Ontology (Amazon)
     - Target Ontology (eBay)
     - Similarity Score
   * - Electronics
     - Computers
     - 0.66
   * - Electronics
     - Notebook
     - 0.41
   * - Laptop
     - Notebook
     - 0.73
   * - GamingLaptop
     - GamingNotebook
     - 0.63
   * - Ultrabook
     - Notebook
     - 0.61

LLM Aligner
---------------------------------------------

For complex or ambiguous cases where previous methods may fall short, we employ a Large Language Model (LLM) to generate potential alignments. The LLM considers broader context and domain knowledge to suggest matches.

.. code-block:: python

    from ontoaligner.encoder import ConceptLLMEncoder
    from ontoaligner.aligner import AutoModelDecoderLLM, ConceptLLMDataset

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    llm_enc = ConceptLLMEncoder()
    src_ctx, tgt_ctx = llm_enc(source=src_onto, target=tgt_onto)

    ds = ConceptLLMDataset(source_onto=src_ctx, target_onto=tgt_ctx)
    dl = DataLoader(ds, batch_size=512, collate_fn=ds.collate_fn)

    llm = AutoModelDecoderLLM(device="cuda", max_length=200, max_new_tokens=5)
    llm.load(path="Qwen/Qwen2-0.5B-Instruct")

    preds = []
    for batch in tqdm(dl):
        seqs = llm.generate(batch["prompts"])
        preds.extend(seqs)


**LLM Predictions Post‑processing**: We process the LLM-generated predictions using a TF-IDF-based label mapper and a logistic regression classifier to determine the most probable alignments.

.. code-block:: python

    from ontoaligner.postprocess import TFIDFLabelMapper, llm_postprocessor
    from sklearn.linear_model import LogisticRegression

    mapper = TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1,1))

    llm_matches = llm_postprocessor(predicts=preds, mapper=mapper, dataset=ds)

.. list-table:: LLM Aligner - Sample LLM Matches
   :header-rows: 1

   * - Source Ontology (Amazon)
     - Target Ontology (eBay)
   * - Electronics
     - Computers
   * - Electronics
     - GamingNotebook
   * - Laptop
     - Computers
   * - Laptop
     - GamingNotebook
   * - Laptop
     - BusinessNotebook
   * - Ultrabook
     - Notebook
   * - Smartphone
     - Computers

------------------------

.. hint::

    - `Sentence-BERT Pretrained Models Guide <https://www.sbert.net/docs/sentence_transformer/pretrained_models.html>`_.
    - Original models: `Sentence Transformers Hugging Face organization <https://huggingface.co/models?library=sentence-transformers&author=sentence-transformers>`_.
    - Community models: `All Sentence Transformer models on Hugging Face <https://huggingface.co/models?library=sentence-transformers>`_.

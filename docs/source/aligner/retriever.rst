Retrieval
==============

Usage
---------------

This tutorial provides a guide to performing ontology alignment using the Retriever based matching model. The process includes loading ontology datasets, generating embeddings, aligning concepts with retrieval models, post-processing the matches, and evaluating the results.

.. raw:: html

   <h4>Step 1: Import the Required Modules</h4>


Start by importing the necessary libraries and modules. These tools will help us process and align the ontologies.

.. code-block:: python

    import json

    # Import modules from OntoAligner
    from ontoaligner import ontology, encoder
    from ontoaligner.utils import metrics, xmlify
    from ontoaligner.aligner import SBERTRetrieval
    from ontoaligner.postprocess import retriever_postprocessor


Here:
- ``SBERTRetrieval``: Pre-trained retrieval model for semantic matching were you can load any sentence-transformer model and use it for matching.
- ``retriever_postprocessor``: Refines matchings for better accuracy.



.. raw:: html

   <h4>Step 2: Initialize, Parse, and Encode Ontology</h4>


Define the ontology alignment task using the provided datasets and then load the ontologies and refrences.

.. code-block:: python

    task = MaterialInformationMatOntoOMDataset()
    print("Test Task:", task)

    dataset = task.collect(
        source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="assets/MI-MatOnto/matchings.xml"
    )

    # Initialize the encoder model and encode the dataset.
    encoder_model = encoder.ConceptParentLightweightEncoder()
    encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])


.. note::
    For retrieval models the ``LightweightEncoder`` encoders are good to use.


.. raw:: html

   <h4>Step 3: Set Up the Retrieval Model and do the Matching</h4>

Configure the retrieval model to align the source and target ontologies using semantic similarity. The `SBERTRetrieval` model leverages a pre-trained transformer for this task.

.. code-block:: python

    # Initialize retrieval model
    model = SBERTRetrieval(device='cpu', top_k=10)
    model.load(path="all-MiniLM-L6-v2")

    # Generate matchings
    matchings = model.generate(input_data=encoder_output)

The retrieval model computes semantic similarities between source and target embeddings, predicting potential alignments.

.. raw:: html

   <h4>Step 4: Post-process and Evaluate the Matchings</h4>


Refine the predicted matchings using the `retriever_postprocessor`. Postprocessing improves alignment quality by filtering or adjusting the results.

.. code-block:: python

    # Post-process matchings
    matchings = retriever_postprocessor(matchings)

    # Evaluate matchings
    evaluation = metrics.evaluation_report(
        predicts=matchings,
        references=dataset['reference']
    )

    # Print evaluation report
    print("Evaluation Report:", json.dumps(evaluation, indent=4))



.. raw:: html

   <h4>Step 5: Export Matchings</h4>


Save the matchings in both XML and JSON formats for further analysis or use. For convert matchings to XML format we use ``xmlify`` utility.

.. code-block:: python

    # Export matchings to XML
    xml_str = xmlify.xml_alignment_generator(matchings=matchings)
    xml_output_path = "matchings.xml"

    with open(xml_output_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_str)

    print(f"Matchings in XML format have been written to '{xml_output_path}'.")

    # Export matchings to JSON
    json_output_path = "matchings.json"

    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(matchings, json_file, indent=4, ensure_ascii=False)

    print(f"Matchings in JSON format have been written to '{json_output_path}'.")

Transformer Aligner
-----------------------------------


.. sidebar:: 🤗 Sentence-Transformers Pre-trained Models

    `https://huggingface.co/sentence-transformers <https://huggingface.co/sentence-transformers>`_

Transformer-based aligners leverage pretrained models from the `sentence-transformers <https://sbert.net/>`_ library (e.g., `BERT <https://huggingface.co/docs/transformers/en/model_doc/bert>`_, `T5 <https://huggingface.co/docs/transformers/en/model_doc/t5>`_, `Flan-T5 <https://huggingface.co/docs/transformers/en/model_doc/flan-t5>`_, `Nomic-AI <https://huggingface.co/collections/nomic-ai/nomic-embed-v2-67acc40c3aa2865aa8a7d114>`_) to encode ontology concepts into dense vector embeddings. ``SBERTRetrieval`` performs similarity-based matching directly over these embeddings, while ``SVMBERTRetrieval`` extends this approach by training an SVM classifier on embedding pairs to make alignment decisions.

.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - Transformer Aligner
     - Description
     - Link
   * - ``SBERTRetrieval``
     - A transformer based aligner support that uses sentence-transformer based models like BERT, T5, FlanT5, Nomic-AI, and etc.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L40-L47>`__
   * - ``SVMBERTRetrieval``
     - Trains a Support Vector Machine (SVM) classifier on embeddings for probabilistic based ranking.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L180-L187>`__

To use transformer based aligner technique:


.. code-block::

    from ontoaligner.aligner import SBERTRetrieval, SVMBERTRetrieval

    aligner = SBERTRetrieval(device="cpu", top_k=5)
    aligner.load(path="all-MiniLM-L6-v2")
    matchings = aligner.generate(input_data=...)

.. hint::

    Replace ``SBERTRetrieval`` with ``SVMBERTRetrieval`` if you are willing to use SVM-based retriever model.

N-Gram Aligner
-----------------------------------

N-Gram aligners apply traditional information retrieval techniques—such as TF-IDF and BM25—to measure textual similarity between ontology concepts based on term frequency patterns. These methods are efficient, interpretable, and particularly effective when concept labels or definitions contain meaningful lexical cues. Ideal for fast, scalable alignment in lexically rich ontologies.

.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - N-Gram Aligner
     - Description
     - Link
   * - ``TFIDFRetrieval``
     - Represents each concept label using a ``TF-IDF`` vector and retrieves alignments based on cosine similarity.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L50-L112>`__
   * - ``BM25Retrieval``
     - BM25 retrieval model (`Okapi BM25 <http://ethen8181.github.io/machine-learning/search/bm25_intro.html>`_) is a probabilistic information retrieval method.This model is used to estimate class(or document) relevance based on term frequency and inverse class(or document) frequency.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L114-L177>`__


To use n-gram based aligner technique:

.. code-block::

    from ontoaligner.aligner import TFIDFRetrieval, BM25Retrieval

    aligner = TFIDFRetrieval(top_k=5)
    matchings = aligner.generate(input_data=...)

.. hint::

    - There is no need for ``.load()`` at this aligners.
    - Replace ``TFIDFRetrieval`` with ``BM25Retrieval`` if you are willing to use BM25-based retriever model.


OpenAI Aligner
-----------------------

OpenAI aligners utilize state-of-the-art embedding models from OpenAI (e.g., ``text-embedding-ada-002``) to represent ontology concepts as dense semantic vectors. These aligners are well-suited for capturing deep contextual meaning across diverse domains and are especially useful when high-quality alignment is needed but local model hosting is not feasible. The embeddings are generated via OpenAI’s API and require an API key and token usage awareness.

.. sidebar:: OpenAI Embeddings:

    OpenAI offers two powerful third-generation embedding model (denoted by -3 in the model ID). Read the embedding v3 `announcement blog post <https://openai.com/index/new-embedding-models-and-api-updates/>`_ for more details. Usage is priced per input token.
    - ``text-embedding-3-small``
    - ``text-embedding-3-large``
    - ``text-embedding-ada-002``


.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - OpenAI Aligner
     - Description
     - Link
   * - ``AdaRetrieval``
     - This model uses pre-trained embeddings from OpenAI. It is designed to use OpenAI embeddings, fit them, and transform input data into corresponding embeddings.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/retrieval/models.py#L189-L241>`__

To use OpenAI based aligner technique:

.. code-block::

    from ontoaligner.aligner import AdaRetrieval

    aligner = AdaRetrieval(top_k=5, openai_key='...')
    aligner.load(path='text-embedding-3-small')
    matchings = aligner.generate(input_data=...)

.. hint::

    More information on OpenAI embeddings can be found at `OpenAI > Embedding models <https://platform.openai.com/docs/guides/embeddings#embedding-models>`_.

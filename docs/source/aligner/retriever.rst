Retrieval Aligner
====================

This tutorial provides a guide to performing ontology alignment using the Retriever based matching model. The process includes loading ontology datasets, generating embeddings, aligning concepts with retrieval models, post-processing the matches, and evaluating the results.

Step 1: Import the Required Modules
------------------------------------

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


Step 2: Initialize, Parse, and Encode Ontology
-----------------------------------------------

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


Step 3: Set Up the Retrieval Model and do the Matching
--------------------------------------------------------

Configure the retrieval model to align the source and target ontologies using semantic similarity. The `SBERTRetrieval` model leverages a pre-trained transformer for this task.

.. code-block:: python

    # Initialize retrieval model
    model = SBERTRetrieval(device='cpu', top_k=10)
    model.load(path="all-MiniLM-L6-v2")

    # Generate matchings
    matchings = model.generate(input_data=encoder_output)

The retrieval model computes semantic similarities between source and target embeddings, predicting potential alignments.

Step 4: Post-process and Evaluate the Matchings
---------------------------------------------------

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



Step 5: Export Matchings
-------------------------

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

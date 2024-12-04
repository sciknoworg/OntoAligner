Retrieval Matching
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
    from ontoaligner.ontology_matchers import SBERTRetrieval
    from ontoaligner.postprocess import retriever_postprocessor


Here:
- ``SBERTRetrieval``: Pre-trained retrieval model for semantic matching were you can load any sentence-transformer model and use it for matching.
- ``retriever_postprocessor``: Refines matchings for better accuracy.


Step 2: Initialize and Parse the Dataset
-----------------------------------------

Define the ontology alignment task using the provided datasets and then load the ontologies and refrences.

.. code-block:: python

    task = MaterialInformationMatOntoOMDataset()
    print("Test Task:", task)

    dataset = task.collect(
        source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="assets/MI-MatOnto/matchings.xml"
    )


This step creates an object that organizes all the required data files and settings for the matching process and using the ``collect`` function it will load the source, target ontologies, and reference matching files. The ``print`` statement confirms that the task has been initialized successfully.

Step 3: Encode the Ontology Data
-------------------------------------

After loading the dataset, the encoder module processes and restructures the concepts from the source and target ontologies, preparing them as input for the matching model. For retrieval models the ``LightweightEncoder`` models are good to use.

.. code-block:: python

    # Initialize the encoder model
    encoder_model = encoder.ConceptParentLightweightEncoder()

    # Generate embeddings
    encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])


Step 4: Set Up the Retrieval Model
-----------------------------------
Configure the retrieval model to align the source and target ontologies using semantic similarity. The `SBERTRetrieval` model leverages a pre-trained transformer for this task.

.. code-block:: python

    # Initialize retrieval model
    model = SBERTRetrieval(device='cpu', top_k=10)
    model.load(path="all-MiniLM-L6-v2")

    # Generate matchings
    matchings = model.generate(input_data=encoder_output)

The retrieval model computes semantic similarities between source and target embeddings, predicting potential alignments.

Step 6: Post-process the Matchings
-----------------------------------
Refine the predicted matchings using the `retriever_postprocessor`. Postprocessing improves alignment quality by filtering or adjusting the results.

.. code-block:: python

    # Post-process matchings
    matchings = retriever_postprocessor(matchings)


Step 7: Evaluate the Matchings
-------------------------------
.. code-block:: python

    # Evaluate matchings
    evaluation = metrics.evaluation_report(
        predicts=matchings,
        references=dataset['reference']
    )

    # Print evaluation report
    print("Evaluation Report:", json.dumps(evaluation, indent=4))



Step 8: Export Matchings
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

Run All at Once
===============

To execute the entire script, use the following consolidated code block. This script performs ontology dataset collection, encoding, retrieval-based alignment, postprocessing, evaluation, and export.

.. code-block:: python

    import json
    from ontoaligner import ontology, encoder
    from ontoaligner.utils import metrics, xmlify
    from ontoaligner.ontology_matchers import SBERTRetrieval
    from ontoaligner.postprocess import retriever_postprocessor

    task = ontology.MaterialInformationMatOntoOMDataset()
    print("Test Task:", task)

    dataset = task.collect(
        source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="../assets/MI-MatOnto/matchings.xml"
    )

    encoder_model = encoder.ConceptParentLightweightEncoder()
    encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])

    model = SBERTRetrieval(device='cpu', top_k=10)
    model.load(path="all-MiniLM-L6-v2")
    matchings = model.generate(input_data=encoder_output)

    matchings = retriever_postprocessor(matchings)

    evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
    print("Evaluation Report:", json.dumps(evaluation, indent=4))

    xml_str = xmlify.xml_alignment_generator(matchings=matchings)
    with open("matchings.xml", "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_str)

    print("Matchings in XML format have been successfully written to 'matchings.xml'.")

    with open("matchings.json", "w", encoding="utf-8") as json_file:
        json.dump(matchings, json_file, indent=4, ensure_ascii=False)

    print("Matchings in JSON format have been successfully written to 'matchings.json'.")

After running the script, you should see:

1. An evaluation report printed in the console.

2. An XML file named matchings.xml saved in the current directory.

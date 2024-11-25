import json

# Import necessary modules from the 'ontoaligner' library
# The library provides tools for ontology alignment tasks, including dataset management, encoding, retrieval models, and postprocessing.
from ontoaligner import ontology, encoder
from ontoaligner.utils import metrics, xmlify
from ontoaligner.ontology_matchers import SBERTRetrieval  # Other available modules: AdaRetrieval, SVMBERTRetrieval, BM25Retrieval
from ontoaligner.postprocess import retriever_postprocessor

# Step 1: Initialize the ontology matching task
# The task is created using the Material Information Ontology Dataset (MI-MatOntoMDataset),
# which includes source and target ontologies and reference matchings for evaluation.
task = ontology.MaterialInformationMatOntoMDataset()

# Confirm the task initialization by printing its details
print("Test Task:", task)

# Step 2: Collect the ontology dataset
# The dataset includes paths to the source ontology, target ontology, and reference matching files.
dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="../assets/MI-MatOnto/matchings.xml"
)

# Step 3: Initialize the encoder model
# The encoder generates embeddings for concepts in the source and target ontologies.
# Here, the 'ConceptParentLightweightEncoder' is used for lightweight encoding, which can be used here as well.
encoder_model = encoder.ConceptParentLightweightEncoder()

# Encode the source and target ontologies
# The encoder processes the concepts in both ontologies and returns embeddings for further alignment.
encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])

# Step 4: Set up the retrieval model
# The retrieval model aligns the source and target ontologies using semantic similarity techniques.
# 'SBERTRetrieval' is selected with a pre-trained model ('all-MiniLM-L6-v2') for embedding retrieval.
model = SBERTRetrieval(device='cpu', top_k=10)
model.load(path="all-MiniLM-L6-v2")

# Generate ontology matchings
# The retrieval model compares encoded embeddings from the source and target datasets to predict matchings.
matchings = model.generate(input_data=encoder_output)

# Step 5: Post-process the matchings
# Apply the 'retriever_postprocessor' function to refine the predicted matchings.
# Postprocessing helps filter or adjust the matchings for improved alignment quality.
matchings = retriever_postprocessor(matchings)

# Step 6: Evaluate the matchings
# The evaluation report compares the predicted matchings against the reference matchings
# provided in the dataset using metrics such as precision, recall, and F1-score.
evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])

# Print the evaluation report in a human-readable JSON format
print("Evaluation Report:", json.dumps(evaluation, indent=4))

# Step 7: Export matchings in XML format or Json format.

# XML format
# Convert the generated matchings into an XML alignment file using the 'xmlify' utility.
xml_str = xmlify.xml_alignment_generator(matchings=matchings)

# Save the XML alignment to a file for further use or analysis
output_file_path = "matchings.xml"
with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

print(f"Matchings in XML format have been successfully written to '{output_file_path}'.")

# OR you can simply store the matchings which is in dictionary format.
# Ensure matchings is a serializable structure before saving
output_file_path = "matchings.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(matchings, json_file, indent=4, ensure_ascii=False)

print(f"Matchings in JSON formathave been successfully written to '{output_file_path}'.")

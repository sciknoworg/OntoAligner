import json

# Import necessary modules from the 'ontoaligner' library
from ontoaligner import ontology, encoder
from ontoaligner.ontology_matchers import SimpleFuzzySMLightweight
from ontoaligner.utils import metrics, xmlify

# Create a task object for material information ontology matching using a predefined dataset
task = ontology.MaterialInformationMatOntoOMDataset()

# Print the test task information (for debugging or confirmation purposes)
print("Test Task:", task)

# Collect the dataset for ontology matching, including source, target, and reference matching
# The paths to the source ontology, target ontology, and reference matching files are provided
dataset = task.collect(source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
                       target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
                       reference_matching_path="../assets/MI-MatOnto/matchings.xml")

# Initialize the encoder model to map concepts between the source and target ontologies
# 'ConceptParentLightweightEncoder' is used here to generate embeddings for matching
encoder_model = encoder.ConceptParentLightweightEncoder()

# Encode the source and target datasets using the encoder model
# The encoder generates embeddings of the concepts in both the source and target ontologies
encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])

# Initialize the SimpleFuzzySMLightweight ontology matcher
# The matcher uses a fuzzy threshold (0.2) to match concepts between the source and target ontologies
model = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2)

# Generate the matching results using the encoded source and target data
# This step applies the fuzzy matching algorithm to align concepts from the source and target ontologies
matchings = model.generate(input_data=encoder_output)

# Evaluate the matching results using an evaluation report
# The 'metrics.evaluation_report' function compares the predicted matchings against the references of source and target ontologies.
evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])

# Print the evaluation report as a formatted JSON string for easy reading
print("Evaluation Report:", json.dumps(evaluation, indent=4))

# Convert the generated matchings into an XML alignment format using 'xmlify.xml_alignment_generator'
# This will produce an XML string that represents the ontology matchings
xml_str = xmlify.xml_alignment_generator(matchings=matchings)

# Write the XML alignment string to a file
# The file is saved with the name 'matchings.xml' in the current directory
with open("matchings.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

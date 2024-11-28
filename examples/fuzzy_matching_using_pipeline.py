import json

# Import necessary modules from the 'ontoaligner' library
from ontoaligner.encoder import ConceptParentLightweightEncoder
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.ontology_matchers import SimpleFuzzySMLightweight
from ontoaligner import Pipeline

# Initialize the ontology matcher with a fuzzy similarity threshold (0.2)
# This model will match concepts between source and target ontologies using fuzzy string matching
ontology_matcher = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2)

# Define the file paths for the source ontology, target ontology, and reference matching file
source_ontology_path = "../assets/MI-MatOnto/mi_ontology.xml"
target_ontology_path = "../assets/MI-MatOnto/matonto_ontology.xml"
reference_matching_path = "../assets/MI-MatOnto/matchings.xml"

# Create a pipeline object with the ontology matcher and encoder model
# The pipeline integrates multiple steps: ontology matching, encoding, and evaluation
pipe = Pipeline(ontology_matcher=ontology_matcher, om_encoder=ConceptParentLightweightEncoder)

# Use the pipeline to process the ontologies and perform ontology matching
# The matching results are evaluated, and the output is returned as a dictionary
matching_dict = pipe(source_ontology_path=source_ontology_path,
                     target_ontology_path=target_ontology_path,
                     reference_matching_path=reference_matching_path,
                     om_dataset=MaterialInformationMatOntoOMDataset,
                     evaluation=True,
                     return_dict=True)

# The dictionary returned contains various keys:
# - 'dataset-info': Information about the dataset used
# - 'encoder-info': Information about the encoder model
# - 'response-time': Time taken for processing
# - 'generated-output': The generated matchings between the source and target ontologies
# - 'evaluation': Evaluation report comparing the predicted matchings with the reference matchings

# Save the matching results as a JSON file, formatted for easy reading
# The results are written to 'matchings.json' with proper indentation and non-ASCII characters preserved
json.dump(matching_dict, open("matchings.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

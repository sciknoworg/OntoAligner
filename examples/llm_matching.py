# Import necessary libraries
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# Import OntoAligner components for ontology matching
from ontoaligner.encoder import ConceptChildrenLLMEncoder
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.ontology_matchers import AutoModelDecoderLLM, ConceptLLMDataset
from ontoaligner.postprocess import TFIDFLabelMapper
from ontoaligner.postprocess import llm_postprocessor

# Initialize the task for ontology matching
task = MaterialInformationMatOntoOMDataset()

# Display information about the task
print("Test Task:", task)

# Collect datasets for the source ontology, target ontology, and reference matchings
dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",  # Path to the source ontology
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",  # Path to the target ontology
    reference_matching_path="../assets/MI-MatOnto/matchings.xml"  # Path to the reference alignments
)

# Initialize an encoder model to extract embeddings for concepts in the ontologies
encoder_model = ConceptChildrenLLMEncoder()

# Encode the source and target ontologies
source_onto, target_onto = encoder_model(source=dataset['source'], target=dataset['target'])

# Prepare the dataset for LLM-based ontology matching
llm_dataset = ConceptLLMDataset(source_onto=source_onto, target_onto=target_onto)

# Create a DataLoader for batching LLM prompts
dataloader = DataLoader(
    llm_dataset,
    batch_size=2048,  # Batch size for processing prompts
    shuffle=False,
    collate_fn=llm_dataset.collate_fn  # Custom collation function for batching
)

# Initialize an LLM-based model for generating ontology alignments
model = AutoModelDecoderLLM(device='cuda', max_length=300, max_new_tokens=10)

# Load pre-trained model weights
model.load(path="Qwen/Qwen2-0.5B")

# List to store predictions
predictions = []

# Generate predictions for ontology alignments using the model
for batch in tqdm(dataloader):  # Iterate through batched data
    prompts = batch["prompts"]  # Extract prompts from the batch
    sequences = model.generate(prompts)  # Generate predictions using the model
    predictions.extend(sequences)  # Append predictions to the list

# Initialize a mapper for post-processing labels using TF-IDF and logistic regression
# The label_dict default value is:
# label_dict = {
#     "yes":["yes", "correct", "true"],
#     "no":["no", "incorrect", "false"]
# }
mapper = TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1, 1))

# Post-process predictions to generate final matchings. The predicted 'yes' class will be keepen as a final LLM based match.
# Since this is a LLM so there would be no score here.
matchings = llm_postprocessor(predicts=predictions, mapper=mapper, dataset=llm_dataset)

# Evaluate the generated matchings against reference alignments
evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])

# Display the evaluation report
print("Evaluation Report:", json.dumps(evaluation, indent=4))

# Convert matchings to XML format for compatibility with ontology alignment tools
xml_str = xmlify.xml_alignment_generator(matchings=matchings)

# Save matchings in XML format to a file
output_file_path = "matchings.xml"
with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

print(f"Matchings in XML format have been successfully written to '{output_file_path}'.")

# Save matchings in JSON format for further analysis or debugging
output_file_path = "matchings.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(matchings, json_file, indent=4, ensure_ascii=False)

print(f"Matchings in JSON format have been successfully written to '{output_file_path}'.")

# Import necessary modules from the ontoaligner package
from ontoaligner.ontology import GraphTripleOMDataset          # Handles ontology data collection in graph triple format
from ontoaligner.encoder import GraphTripleEncoder             # Encodes graph triple data into model-consumable format
from ontoaligner.aligner import ConvEAligner                   # Alignment model based on ConvE (Knowledge Graph Embedding)
from ontoaligner.postprocess import graph_postprocessor        # Applies post-processing to improve alignment quality
from ontoaligner.utils import metrics, xmlify                  # Utilities for evaluation and XML export

# Step 1: Initialize ontology matching task
task = GraphTripleOMDataset()
task.ontology_name = "Mouse-Human"                             # Assign a name for the ontology matching task
print("task:", task)
# Example output: task: Track: GraphTriple, Source-Target sets: Mouse-Human

# Step 2: Load source, target, and reference ontologies
dataset = task.collect(
    source_ontology_path="assets/mouse-human/source.xml",      # Path to source ontology
    target_ontology_path="assets/mouse-human/target.xml",      # Path to target ontology
    reference_matching_path="assets/mouse-human/reference.xml" # Path to ground-truth reference alignments
)
print("dataset key-values:", dataset.keys())
# Example output: dict_keys(['dataset-info', 'source', 'target', 'reference'])

# Print a sample of the parsed source ontology data
print("Sample source ontology:", dataset['source'][0])

# Step 3: Encode the dataset into a format suitable for the aligner
encoder = GraphTripleEncoder()
encoded_dataset = encoder(**dataset)                           # Transforms raw ontology triples into embedding-compatible format

# Step 4: Define training parameters for the ConvE aligner
kge_params = {
    'device': 'cpu',                  # Device to use ('cpu' or 'cuda')
    'embedding_dim': 300,            # Size of embedding vectors
    'num_epochs': 50,                # Total number of training epochs
    'train_batch_size': 128,         # Batch size for training
    'eval_batch_size': 64,           # Batch size for evaluation
    'num_negs_per_pos': 5,           # Number of negative samples for each positive sample
    'random_seed': 42,               # Seed for reproducibility
}

# Step 5: Initialize and train the aligner model
aligner = ConvEAligner(**kge_params)
matchings = aligner.generate(input_data=encoded_dataset)       # Generate predicted alignments between source and target ontologies

# Step 6: Post-process the predicted matchings
processed_matchings = graph_postprocessor(predicts=matchings, threshold=0.5)
# Filters matchings using a similarity threshold (e.g., 0.5)

# Step 7: Evaluate matchings before and after post-processing
evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
print("Matching Evaluation Report:\n", evaluation)

evaluation = metrics.evaluation_report(predicts=processed_matchings, references=dataset['reference'])
print("Matching Evaluation Report -- after post-processing:\n", evaluation)

# Step 8: Convert processed matchings to XML format and save to file
xml_str = xmlify.xml_alignment_generator(matchings=processed_matchings)
with open("matchings.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

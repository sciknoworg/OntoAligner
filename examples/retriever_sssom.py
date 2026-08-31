import json

# Import necessary modules from the 'ontoaligner' library
# The library provides tools for ontology alignment tasks, including dataset management, encoding, retrieval models, and postprocessing.
from ontoaligner import ontology, encoder
from ontoaligner.utils import metrics
from ontoaligner.utils.sssom_generator import sssom_alignment_generator
from ontoaligner.aligner import SBERTRetrieval  # Other available modules: AdaRetrieval, SVMBERTRetrieval, BM25Retrieval
from ontoaligner.postprocess import retriever_postprocessor


# Step 1: Initialize the ontology matching task
# The task is created using the Mouse-Human ontology matching dataset,
# which includes source and target anatomy ontologies and reference matchings for evaluation.
task = ontology.MouseHumanOMDataset()

# Confirm the task initialization by printing its details
print("Test Task:", task)


# Step 2: Collect the ontology dataset
# The dataset includes paths to the source ontology, target ontology, and reference matching files.
dataset = task.collect(
    source_ontology_path="assets/mouse-human/source.xml",
    target_ontology_path="assets/mouse-human/target.xml",
    reference_matching_path="assets/mouse-human/reference.xml",
)


# Step 3: Initialize the encoder model
# The encoder prepares concept representations from the source and target ontologies.
# Here, the 'ConceptParentLightweightEncoder' is used for lightweight encoding, which can be used here as well.
encoder_model = encoder.ConceptParentLightweightEncoder()

# Encode the source and target ontologies
# The encoder processes the concepts in both ontologies and returns embeddings for further alignment.
encoder_output = encoder_model(
    source=dataset["source"],
    target=dataset["target"],
)


# Step 4: Set up the retrieval model
# The retrieval model aligns the source and target ontologies using semantic similarity techniques.
# 'SBERTRetrieval' is selected with a pre-trained model ('all-MiniLM-L6-v2') for embedding retrieval.
model = SBERTRetrieval(
    device="cpu",
    top_k=1,
)

model.load(
    path="all-MiniLM-L6-v2",
)

# Generate ontology matchings
# The retrieval model compares encoded embeddings from the source and target datasets to predict matchings.
matchings = model.generate(
    input_data=encoder_output,
)


# Step 5: Post-process the matchings
# Apply the 'retriever_postprocessor' function to refine the predicted matchings.
# Postprocessing helps filter or adjust the matchings for improved alignment quality.
threshold = 0.2

matchings = retriever_postprocessor(
    predicts=matchings,
    threshold=threshold,
)


# Step 6: Evaluate the matchings
# The evaluation report compares the predicted matchings against the reference matchings
# provided in the dataset using metrics such as precision, recall, and F1-score.
evaluation = metrics.evaluation_report(
    predicts=matchings,
    references=dataset["reference"],
)

# Print the evaluation report in a human-readable JSON format
print(
    "Evaluation Report:",
    json.dumps(evaluation, indent=4),
)


# Step 7: Export matchings in SSSOM format
# Convert the generated matchings into an SSSOM TSV alignment file using
# the 'sssom_alignment_generator' utility.
sssom_str = sssom_alignment_generator(
    matchings=matchings,
    source=dataset["source"],
    target=dataset["target"],
    predicate_id="owl:equivalentClass",
    mapping_set_metadata={
        "mapping_set_id": "https://example.org/mappings/mouse-human-sbert",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "subject_type": "owl:Class",
        "object_type": "owl:Class",
    },
    # Explicit CURIE maps must include all used prefixes; if omitted, Bioregistry is used as fallback.
    curie_map={
        "mouse": "http://mouse.owl#",
        "human": "http://human.owl#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "semapv": "https://w3id.org/semapv/vocab/",
    },
    aligner=model,
    postprocessor=retriever_postprocessor,
    postprocessor_params={
        "threshold": threshold,
    },
    # If you don't want per-matching aligner/postprocessor metadata (e.g. similarity
    # scores or mapping_justification inferred from the aligner), set
    # include_aligner_metadata=False to produce a minimal SSSOM output.
    include_aligner_metadata=False,
)


# Save the SSSOM alignment to a file for further use or analysis
output_file_path = "mouse-human-sbert.sssom.tsv"

with open(output_file_path,"w",encoding="utf-8",newline="",) as sssom_file:
    sssom_file.write(sssom_str)

print(
    f"Matchings in SSSOM format have been successfully written to "
    f"'{output_file_path}'."
)

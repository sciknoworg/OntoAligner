import json
import torch
# import os

from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.encoder import ConceptParentLightweightEncoder
from ontoaligner.aligner import (
    SBERTRetrieval,
    CrossEncoderReranking,
    # CohereReranking,
)
from ontoaligner.postprocess import retriever_postprocessor
from ontoaligner import AlignerPipeline


# Step 1: Initialize the dataset object
task = MaterialInformationMatOntoOMDataset()
print("Test Task:", task)

# Step 2: Load source and target ontologies with reference matchings
dataset = task.collect(
    source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="assets/MI-MatOnto/matchings.xml",
)

# Step 3: Select the runtime device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 4: Define the aligner pipeline with reranking
reranker = CrossEncoderReranking(
    device=device,
    top_k=5,
    normalize_score="sigmoid",
)

reranker_load_params = {
    "path": "cross-encoder/ms-marco-MiniLM-L6-v2",
}

# To use Cohere reranking instead of CrossEncoderReranking, replace the
# reranker and reranker_load_params blocks above with:
#
# reranker = CohereReranking(
#     cohere_key=os.environ["COHERE_API_KEY"],
#     top_k=5,
#     normalize_score="none",
# )
#
# reranker_load_params = {
#     "path": "rerank-v3.5",
# }

aligner_pipeline = AlignerPipeline(
    encoder=ConceptParentLightweightEncoder(),
    aligner=SBERTRetrieval(
        device=device,
        top_k=10,
    ),
    om_dataset=dataset,
    load_params={
        "path": "all-MiniLM-L6-v2",
    },
    reranker=reranker,
    reranker_load_params=reranker_load_params,
    postprocessor=retriever_postprocessor,
    postprocessor_params={
        "threshold": 0.5,
    },
)

# Step 5: Generate predictions
matchings = aligner_pipeline.generate()

# Step 6: Evaluate predictions
evaluation = metrics.evaluation_report(
    predicts=matchings,
    references=dataset["reference"],
)

print("\nAligner Pipeline with Reranking Evaluation Report:")
print(json.dumps(evaluation, indent=4))

# Step 7: Save XML output
xml_str = xmlify.xml_alignment_generator(matchings=matchings)

with open("aligner_pipeline_reranking_matchings.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

print("Saved XML: aligner_pipeline_reranking_matchings.xml")
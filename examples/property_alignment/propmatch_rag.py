import json

from ontoaligner.ontology import PropertyOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.aligner import FalconLLMBERTRetrieverRAG
from ontoaligner.encoder import PropertyFullTextRAGEncoder
from ontoaligner.postprocess import rag_hybrid_postprocessor

# Step 1: Initialize the property ontology matching task
task = PropertyOMDataset()
print("Property Matching Task:", task)

# Step 2: Collect source ontology, target ontology, and reference property alignments
dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="../assets/MI-MatOnto/property_matchings.xml",
)

# Step 3: Initialize the property RAG encoder
# This encoder should use:
#   retrieval_encoder = PropMatchEncoder
#   llm_encoder = "PropertyFullTextRAGDataset"
encoder_model = PropertyFullTextRAGEncoder()

# Step 4: Encode the property ontologies
encoded_ontology = encoder_model(
    source=dataset["source"],
    target=dataset["target"],
    reference=dataset["reference"],
)

# Step 5: Define model configuration
retriever_config = {
    "device": "cpu",
    "top_k": 5,
    "threshold": 0.1,
}
llm_config = {
    "device": "cpu",
    "max_length": 300,
    "max_new_tokens": 5,
    "huggingface_access_token": "",
    "device_map": "auto",
    "batch_size": 8,
    "answer_set": {
        "yes": ["yes", "correct", "true", "positive", "valid"],
        "no": ["no", "incorrect", "false", "negative", "invalid"],
    }
}


# Step 6: Initialize the normal RAG model
model = FalconLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)

# Step 7: Load small LLM and retriever model
model.load(
    llm_path="Qwen/Qwen2.5-0.5B-Instruct",
    ir_path="all-MiniLM-L6-v2",
)

# Step 8: Generate property matching predictions
predicts = model.generate(input_data=encoded_ontology)

print(predicts)
# Step 9: Apply hybrid postprocessing
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(
    predicts=predicts,
    ir_score_threshold=0.4,
    llm_confidence_th=0.5,
)

# Step 10: Evaluate property matchings
evaluation = metrics.evaluation_report(
    predicts=hybrid_matchings,
    references=dataset["reference"],
)

print("Property Hybrid Matching Evaluation Report:")
print(json.dumps(evaluation, indent=4))

# Step 11: Print hybrid postprocessing configuration
print("Property Hybrid Matching Obtained Configuration:")
print(hybrid_configs)

# Step 12: Convert final property matchings to XML
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

# Step 13: Save XML output
output_file_path = "property_matchings.xml"

with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

print(f"Saved property matchings to: {output_file_path}")
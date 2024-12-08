# Import required libraries and modules
import json
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.ontology_matchers import MistralLLMBERTRetrieverRAG
from ontoaligner.encoder import ConceptParentRAGEncoder
from ontoaligner.postprocess import rag_hybrid_postprocessor, rag_heuristic_postprocessor

# Step 1: Initialize the dataset object for MaterialInformation MatOnto dataset
task = MaterialInformationMatOntoOMDataset()

# Print the task to verify initialization
print("Test Task:", task)

# Step 2: Load source and target ontologies along with reference matchings
dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",  # Path to the source ontology
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",  # Path to the target ontology
    reference_matching_path="../assets/MI-MatOnto/matchings.xml"  # Path to the reference matching file
)

# Step 3: Encode the source and target ontologies
# Use ConceptParentRAGEncoder to encode the ontologies
encoder_model = ConceptParentRAGEncoder()
encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

# Step 4: Define configuration for retriever and LLM
retriever_config = {
    "device": 'cuda',  # Use GPU for retrieval
    "top_k": 5,  # Number of top candidates to retrieve
    "threshold": 0.1,  # Confidence threshold for retrieval
    # Uncomment and set your OpenAI key if using OpenAI models
    # "openai_key": ""
}

llm_config = {
    "device": "cuda",  # Use GPU for LLM
    "max_length": 300,  # Maximum length for LLM input
    "max_new_tokens": 10,  # Maximum number of new tokens generated by LLM
    "huggingface_access_token": "",  # Access token for Huggingface models (if required)
    "device_map": 'balanced',  # Distribute LLM model across devices
    "batch_size": 15,  # Batch size for processing
    "answer_set": {  # Define answer mapping for yes/no responses
        "yes": ["yes", "correct", "true", "positive", "valid"],
        "no": ["no", "incorrect", "false", "negative", "invalid"]
    }
    # Uncomment and set your OpenAI key if using OpenAI models
    # "openai_key": ""
}

# Step 5: Initialize the RAG-based ontology matcher
model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
model.load(llm_path = "mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")

# Generate predictions using the matcher
predicts = model.generate(input_data=encoded_ontology)

# Step 6: Apply heuristic postprocessing
# Automatically find thresholds for retrieval and LLM based on top-k candidates
heuristic_matchings, heuristic_configs = rag_heuristic_postprocessor(
    predicts=predicts,
    topk_confidence_ratio=3,  # Top-k confidence ratio for heuristic postprocessing
    topk_confidence_score=3  # Top-k confidence score for heuristic postprocessing
)

# Evaluate heuristic matchings against the reference
evaluation = metrics.evaluation_report(predicts=heuristic_matchings, references=dataset['reference'])
print("Heuristic Matching Evaluation Report:", json.dumps(evaluation, indent=4))
print("Heuristic Matching Obtained Configuration:", heuristic_configs)

# Step 7: Apply hybrid postprocessing
# Filter matchings based on fixed thresholds for retrieval and LLM confidence
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(
    predicts=predicts,
    ir_score_threshold=0.1,  # IR score threshold for hybrid postprocessing
    llm_confidence_th=0.8  # LLM confidence threshold for hybrid postprocessing
)

# Evaluate hybrid matchings against the reference
evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))
print("Hybrid Matching Obtained Configuration:", hybrid_configs)

# Step 8: Convert matchings to XML format
# Use XMLify utility to generate XML representation of the matchings
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

# Save the XML representation to a file
output_file_path = "matchings.xml"
with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)
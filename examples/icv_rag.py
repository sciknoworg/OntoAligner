import json
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.aligner import FalconLLMBERTRetrieverICVRAG
from ontoaligner.encoder import ConceptRAGEncoder
from ontoaligner.postprocess import rag_hybrid_postprocessor

# Step 1: Initialize the Ontology Matching Task
# The MaterialInformationMatOntoOMDataset object is created to start the ontology matching task
task = MaterialInformationMatOntoOMDataset()
print("Test Task:", task)

# Step 2: Collect the dataset for ontology matching
# This collects the source ontology, target ontology, and reference alignments required for the matching process
dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",  # Path to the source ontology file
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",  # Path to the target ontology file
    reference_matching_path="../assets/MI-MatOnto/matchings.xml"  # Path to the reference alignments
)

# Step 3: Initialize the ConceptRAGEncoder model
# This encoder will process the source and target ontologies for the matching process
encoder_model = ConceptRAGEncoder()

# Step 4: Encode the ontologies
# This encodes the source, target, and reference ontologies into a format that can be used by the model
encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'], reference=dataset['reference'])

# Step 5: Define model configuration for FalconLLMBERTRetrieverICVRAG
# This includes the device settings, retrieval configuration, and LLM configuration for the matching model
config = {
    "retriever_config": {
        "device": 'cuda',  # Specify the device for computation
        "top_k": 5,  # Number of top candidates to retrieve
        "threshold": 0.1,  # Threshold for IR scores
        # "openai_key": ""  # Uncomment to use OpenAI models
    },
    "llm_config": {
        "device": "cuda",  # Specify the device for computation
        "max_length": 300,  # Max length for LLM input
        "max_new_tokens": 10,  # Max new tokens for generation
        "huggingface_access_token": "",  # Huggingface access token for restricted models
        "device_map": 'balanced',  # Device mapping strategy
        "batch_size": 32,  # Batch size for inference
        "answer_set": {
            "yes": ["yes", "correct", "true", "positive", "valid"],
            "no": ["no", "incorrect", "false", "negative", "invalid"]
        }
        # "openai_key": ""  # Uncomment to use OpenAI models
    }
}

# Step 6: Initialize the Matching Model (FalconLLMBERTRetrieverICVRAG)
# This step creates an instance of the FalconLLMBERTRetrieverICVRAG model with the configuration provided
model = FalconLLMBERTRetrieverICVRAG(**config)

# Step 7: Load pre-trained models (LLM and IR models)
# Loads the pre-trained Falcon LLM and information retrieval models
model.load(
    llm_path="tiiuae/falcon-7b",  # Path to the pre-trained LLM
    ir_path="all-MiniLM-L6-v2"  # Path to the IR model
)

# Step 8: Generate predictions using the model
# This generates predictions for the ontology matching task using the encoded data
predicts = model.generate(input_data=encoded_ontology)

# Step 9: Post-process the predictions using a hybrid approach
# This applies a post-processing step to refine the predictions based on IR and LLM confidence thresholds
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(
    predicts=predicts,
    ir_score_threshold=0.4,  # IR score threshold for filtering
    llm_confidence_th=0.5  # LLM confidence threshold for filtering
)

# Step 10: Evaluate the hybrid matchings
# This generates an evaluation report comparing the predicted matchings to the reference matchings
evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))

# Step 11: Print the hybrid matching configuration
# This outputs the configuration used for hybrid matching
print("Hybrid Matching Obtained Configuration:", hybrid_configs)

# Step 12: Convert the hybrid matchings to XML format
# This generates an XML representation of the final ontology matchings
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

# Step 13: Save the XML output to a file
# The resulting XML string is saved to an XML file for further use
output_file_path = "matchings.xml"
with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

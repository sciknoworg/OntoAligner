# Import necessary modules
import json
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.aligner import MistralLLMBERTRetrieverFSRAG
from ontoaligner.encoder import ConceptParentFewShotEncoder
from ontoaligner.postprocess import rag_hybrid_postprocessor

# Step 1: Initialize the task for ontology alignment
task = MaterialInformationMatOntoOMDataset()
print("Test Task:", task)

# Step 2: Collect the dataset
dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",  # Path to the source ontology file
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",  # Path to the target ontology file
    reference_matching_path="../assets/MI-MatOnto/matchings.xml"  # Path to the reference alignments
)

# Step 3: Encode the ontology concepts
encoder_model = ConceptParentFewShotEncoder()
encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'], reference=dataset['reference'])

# Step 4: Configure the RAG model
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

# Step 5: Initialize and load the RAG model
model = MistralLLMBERTRetrieverFSRAG(positive_ratio=0.7, n_shots=5, **config)
model.load(
    llm_path="mistralai/Mistral-7B-v0.3",  # Path to the pre-trained LLM
    ir_path="all-MiniLM-L6-v2"  # Path to the IR model
)

# Step 6: Generate predictions using the model
predicts = model.generate(input_data=encoded_ontology)

# Step 7: Hybrid postprocessing of matchings
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(
    predicts=predicts,
    ir_score_threshold=0.3,  # IR score threshold for filtering
    llm_confidence_th=0.5  # LLM confidence threshold for filtering
)

# Step 8: Evaluate the hybrid matchings
evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))
print("Hybrid Matching Obtained Configuration:", hybrid_configs)


# Step 9: Generate XML output for the matchings
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)

# Step 10: Save the XML output to a file
output_file_path = "matchings.xml"
with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

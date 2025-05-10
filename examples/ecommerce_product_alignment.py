# Import necessary modules
from ontoaligner.ontology import GenericOntology
from ontoaligner import encoder
from ontoaligner.ontology_matchers import SimpleFuzzySMLightweight, SBERTRetrieval
from ontoaligner.postprocess import retriever_postprocessor
from ontoaligner.encoder import ConceptLLMEncoder
from ontoaligner.ontology_matchers import AutoModelDecoderLLM, ConceptLLMDataset
from ontoaligner.postprocess import TFIDFLabelMapper, llm_postprocessor

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# Step 1: Load and parse ontologies
ontology = GenericOntology()
parsed_ontology1 = ontology.parse("amazon.owl")  # Path to source ontology
parsed_ontology2 = ontology.parse("ebay.owl")     # Path to target ontology

# Step 2: Encode ontologies using a lightweight parent-based encoder
encoder_model = encoder.ConceptParentLightweightEncoder()
encoder_output = encoder_model(source=parsed_ontology1, target=parsed_ontology2)

# Step 3: Match using Simple Fuzzy String Matcher (Lightweight)
fuzzy_model = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.4)  # Set fuzzy string threshold
fuzzy_matchings = fuzzy_model.generate(input_data=encoder_output)

# Step 4: Print fuzzy matcher results
print("\n=== SimpleFuzzySMLightweight Matchings ===")
for match in fuzzy_matchings:
    print(f"Source: {match['source']}, Target: {match['target']}, Score: {match['score']}")

# Step 5: Match using SBERT Retriever
sbert_model = SBERTRetrieval(device="cpu", top_k=3)  # Top 3 candidates using SBERT
sbert_model.load(path="all-MiniLM-L6-v2")  # Load pre-trained model
sbert_matchings = sbert_model.generate(input_data=encoder_output)

# Step 6: Print SBERT matcher results
print("\n=== SBERTRetrieval Matchings ===")
for match in sbert_matchings:
    print(f"Source: {match['source']}, Target: {match['target-cands']}, Score: {match['score-cands']}")

# Step 7: Postprocess SBERT matchings (e.g., filter duplicates, normalize)
sbert_matchings = retriever_postprocessor(sbert_matchings)

# Step 8: Print postprocessed SBERT matchings
print("\n=== Post-Processed SBERT Matchings ===")
for match in sbert_matchings:
    print(f"Source: {match['source']}, Target: {match['target']}, Score: {match['score']}")

# Step 9: Encode ontologies for LLM-based matching
llm_encoder = ConceptLLMEncoder()
source_onto, target_onto = llm_encoder(source=parsed_ontology1, target=parsed_ontology2)

# Step 10: Prepare dataset for LLM decoder
llm_dataset = ConceptLLMDataset(source_onto=source_onto, target_onto=target_onto)

# Step 11: Create a DataLoader for batching LLM prompts
dataloader = DataLoader(
    llm_dataset,
    batch_size=2048,  # Batch size for LLM inference
    shuffle=False,
    collate_fn=llm_dataset.collate_fn
)

# Step 12: Load and configure the LLM decoder
llm_model = AutoModelDecoderLLM(device="cuda", max_length=300, max_new_tokens=10)
llm_model.load(path="Qwen/Qwen2-0.5B")  # Load a small Qwen model

# Step 13: Generate predictions using LLM decoder
predictions = []
for batch in tqdm(dataloader, desc="Generating with LLM"):
    prompts = batch["prompts"]
    sequences = llm_model.generate(prompts)
    predictions.extend(sequences)

# Step 14: Postprocess LLM predictions using TF-IDF label mapper
mapper = TFIDFLabelMapper(
    classifier=LogisticRegression(),  # Classifier for label similarity
    ngram_range=(1, 1)                # Use unigram TF-IDF
)
llm_matchings = llm_postprocessor(predicts=predictions, mapper=mapper, dataset=llm_dataset)

# Step 15: Print final LLM-based matchings
print("\n=== LLM Matchings ===")
for match in llm_matchings:
    print(f"Source: {match['source']}, Target: {match['target']}, Score: {match['score']}")

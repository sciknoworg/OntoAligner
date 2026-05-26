import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from ontoaligner.ontology import PropertyOMDataset
from ontoaligner.encoder import PropMatchEncoder
from ontoaligner.aligner import AutoModelDecoderLLM
from ontoaligner.aligner import PropertyFullTextLLMDataset

from ontoaligner.postprocess import TFIDFLabelMapper, llm_postprocessor
from ontoaligner.utils import metrics, xmlify

# ---------------------------------------------------------
# Step 1: Initialize the property ontology matching task
# ---------------------------------------------------------
task = PropertyOMDataset()

print("Property Matching Task:", task)

# ---------------------------------------------------------
# Step 2: Collect source ontology, target ontology, and references
# ---------------------------------------------------------
dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="../assets/MI-MatOnto/property_matchings.xml",
)

# ---------------------------------------------------------
# Step 3: Encode properties
# ---------------------------------------------------------
# PropMatchEncoder should produce property dictionaries containing:
# iri, label, domain, range, inverse
#
# These fields are used by PropertyFullTextLLMDataset.
encoder_model = PropMatchEncoder()

source_onto, target_onto = encoder_model(
    source=dataset["source"],
    target=dataset["target"],
)
# ---------------------------------------------------------
# Step 4: Prepare property LLM dataset
# ---------------------------------------------------------
llm_dataset = PropertyFullTextLLMDataset(
    source_onto=source_onto,
    target_onto=target_onto,
)
print("Number of property pairs:", len(llm_dataset))

# ---------------------------------------------------------
# Step 5: Create DataLoader
# ---------------------------------------------------------
dataloader = DataLoader(
    llm_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=llm_dataset.collate_fn,
)

# ---------------------------------------------------------
# Step 6: Initialize LLM model
# ---------------------------------------------------------
model = AutoModelDecoderLLM(
    device="cpu",        # Use "cpu" if GPU is not available
    max_length=300,
    max_new_tokens=10,
)

# ---------------------------------------------------------
# Step 7: Load LLM
# ---------------------------------------------------------
model.load(
    path="Qwen/Qwen2.5-0.5B-Instruct"
)

# ---------------------------------------------------------
# Step 8: Generate LLM predictions
# ---------------------------------------------------------
predictions = []

for batch in tqdm(dataloader):
    prompts = batch["prompts"]
    sequences = model.generate(prompts)
    predictions.extend(sequences)

print("Number of predictions:", len(predictions))


# ---------------------------------------------------------
# Step 9: Map LLM outputs to yes/no
# ---------------------------------------------------------
label_dict = {
    "yes": ["yes", "correct", "true", "positive", "valid"],
    "no": ["no", "incorrect", "false", "negative", "invalid"],
}

mapper = TFIDFLabelMapper(
    classifier=LogisticRegression(),
    ngram_range=(1, 1),
    label_dict=label_dict,
)

# ---------------------------------------------------------
# Step 10: Post-process LLM predictions
# ---------------------------------------------------------
# llm_postprocessor keeps predicted "yes" pairs as final matchings.
matchings = llm_postprocessor(
    predicts=predictions,
    mapper=mapper,
    dataset=llm_dataset,
)

# ---------------------------------------------------------
# Step 11: Evaluate property matchings
# ---------------------------------------------------------
evaluation = metrics.evaluation_report(
    predicts=matchings,
    references=dataset["reference"],
)
print("Property LLM Matching Evaluation Report:")
print(json.dumps(evaluation, indent=4))

# ---------------------------------------------------------
# Step 12: Save XML matchings
# ---------------------------------------------------------
xml_str = xmlify.xml_alignment_generator(matchings=matchings)
xml_output_file = "property_llm_matchings.xml"
with open(xml_output_file, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)
print(f"Saved property LLM matchings XML to: {xml_output_file}")

# ---------------------------------------------------------
# Step 13: Save JSON matchings
# ---------------------------------------------------------
json_output_file = "property_llm_matchings.json"

with open(json_output_file, "w", encoding="utf-8") as json_file:
    json.dump(matchings, json_file, indent=4, ensure_ascii=False)

print(f"Saved property LLM matchings JSON to: {json_output_file}")
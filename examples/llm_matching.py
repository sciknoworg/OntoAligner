import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from ontoaligner.encoder import ConceptChildrenLLMEncoder
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.ontology_matchers import AutoModelDecoderLLM, ConceptLLMDataset
from ontoaligner.postprocess import TFIDFLabelMapper
from ontoaligner.postprocess import llm_postprocessor

task = MaterialInformationMatOntoOMDataset()

print("Test Task:", task)

dataset = task.collect(
    source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="../assets/MI-MatOnto/matchings.xml"
)

encoder_model = ConceptChildrenLLMEncoder()
source_onto, target_onto = encoder_model(source=dataset['source'], target=dataset['target'])

llm_dataset = ConceptLLMDataset(source_onto=source_onto, target_onto=target_onto)
dataloader = DataLoader(llm_dataset, batch_size=2048, shuffle=False, collate_fn=llm_dataset.collate_fn)

model = AutoModelDecoderLLM(device='cuda', max_length=300, max_new_tokens=10)
model.load(path="Qwen/Qwen2-0.5B")

predictions = []
for batch in tqdm(dataloader):
    prompts = batch["prompts"]
    sequences = model.generate(prompts)
    predictions.extend(sequences)

# label_dict = {
#     "yes":["yes", "correct", "true"],
#     "no":["no", "incorrect", "false"]
# }
mapper = TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1, 1))

matchings = llm_postprocessor(predicts=predictions, mapper=mapper, dataset=llm_dataset)

evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])

print("Evaluation Report:", json.dumps(evaluation, indent=4))

xml_str = xmlify.xml_alignment_generator(matchings=matchings)

output_file_path = "matchings.xml"
with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

print(f"Matchings in XML format have been successfully written to '{output_file_path}'.")

output_file_path = "matchings.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(matchings, json_file, indent=4, ensure_ascii=False)

print(f"Matchings in JSON formathave been successfully written to '{output_file_path}'.")

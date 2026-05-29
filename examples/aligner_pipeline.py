import json

from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.encoder import ConceptParentLightweightEncoder
from ontoaligner.aligner import SimpleFuzzySMLightweight
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

# Step 3: Define the aligner pipeline
aligner_pipeline = AlignerPipeline(
    encoder=ConceptParentLightweightEncoder(),
    aligner=SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2),
    om_dataset=dataset,
)

# Step 4: Generate predictions
matchings = aligner_pipeline.generate()

# Step 5: Evaluate predictions
evaluation = metrics.evaluation_report(
    predicts=matchings,
    references=dataset["reference"],
)

print("\n Aligner Pipeline Evaluation Report:")
print(json.dumps(evaluation, indent=4))

# Step 6: Save XML output
xml_str = xmlify.xml_alignment_generator(matchings=matchings)

with open("aligner_pipeline_matchings.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

print("Saved XML: aligner_pipeline_matchings.xml")

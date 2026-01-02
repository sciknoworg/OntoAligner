import json
from ontoaligner import ontology, encoder
from ontoaligner.aligner import SimpleFuzzySMLightweight
from ontoaligner.utils import metrics

task = ontology.PropertyOMDataset(ontology.ProcessingStrategy.MOST_COMMON_PAIRS)
dataset = task.collect(source_ontology_path="assets/cmt-conference/source.xml",
                       target_ontology_path="assets/cmt-conference/target.xml",
                       reference_matching_path="assets/cmt-conference/reference.xml")

encoder_model = encoder.PropertyEncoder()
encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])

aligner = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.8)
matchings = aligner.generate(input_data=encoder_output)

evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
print("Evaluation Report:", json.dumps(evaluation, indent=4))

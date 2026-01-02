import json
from ontoaligner import ontology, encoder
from ontoaligner.aligner import SBERTRetrieval
from ontoaligner.postprocess import retriever_postprocessor
from ontoaligner.utils import metrics

task = ontology.PropertyOMDataset(ontology.ProcessingStrategy.MOST_COMMON_PAIRS)
dataset = task.collect(source_ontology_path="assets/cmt-conference/source.xml",
                       target_ontology_path="assets/cmt-conference/target.xml",
                       reference_matching_path="assets/cmt-conference/reference.xml")

encoder_model = encoder.PropertyEncoder()
encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])

aligner = SBERTRetrieval(device='cpu', top_k=5)
aligner.load(path="all-MiniLM-L6-v2")

matchings = aligner.generate(input_data=encoder_output)
matchings = retriever_postprocessor(matchings)

evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
print("Evaluation Report:", json.dumps(evaluation, indent=4))

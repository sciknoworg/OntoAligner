import json
from ontoaligner import ontology, encoder
from  ontoaligner.aligner import PropMatchAligner
from ontoaligner.utils import metrics

task = ontology.PropertyOMDataset(ontology.ProcessingStrategy.NONE)
# task = ontology.PropertyOMDataset(ontology.ProcessingStrategy.MOST_COMMON_DOMAIN_RANGE)

dataset = task.collect(source_ontology_path="assets/cmt-conference/source.xml",
                       target_ontology_path="assets/cmt-conference/target.xml",
                       reference_matching_path="assets/cmt-conference/reference.xml")

encoder_model = encoder.PropMatchEncoder()

encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])

aligner = PropMatchAligner(fmt='glove', threshold=0.0, steps=1, sim_weight=[0, 1, 2])
aligner.load(wordembedding_path='glove.6B/glove.6B.50d.txt', sentence_transformer_id='all-MiniLM-L12-v2')

matchings = aligner.generate(input_data=encoder_output)

evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
print("Evaluation Report:", json.dumps(evaluation, indent=4))

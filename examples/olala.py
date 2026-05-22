# Run OLaLa in OntoAligner

import json
from ontoaligner.ontology import OLaLaOMDataset
from ontoaligner.encoder import OLaLaEncoder
from ontoaligner.aligner.olala import (OLaLaSBERTRetrieval,
                                       OLaLaLLMAligner,
                                       OLaLaHighPrecisionMatcher,
                                       OLaLaAligner)
from ontoaligner.aligner.olala.postprocessor import olala_postprocessor
from ontoaligner.utils import metrics, xmlify


# 1. Load task and ontologies
task = OLaLaOMDataset()
print("Test Task:", task)
dataset = task.collect(
    source_ontology_path="../assets/source.owl",
    target_ontology_path="../assets/target.owl",
    reference_matching_path="../assets/reference.xml",
)
# 2. Encode ontologies
encoder_model = OLaLaEncoder()

encoded_ontology = encoder_model(
    source=dataset["source"],
    target=dataset["target"],
)

# 3. SBERT candidate generation
retriever = OLaLaSBERTRetrieval(
    device="cuda",
    top_k=5,
    both_directions=True,
    topk_per_resource=True,
)

# 4. LLM binary verification
llm_aligner = OLaLaLLMAligner(
    device="cuda",
    max_new_tokens=10,
    temperature=0.0,
    truncation=True,
    max_length=2048,
    padding=True,
    loading_arguments={
        "device_map": "auto",
        "torch_dtype": "torch.float16",
    },
)

# 5. High-precision matcher
hp_aligner = OLaLaHighPrecisionMatcher(confidence=1.0)

# 6. OLaLa alignments
olala = OLaLaAligner(retriever=retriever,
                     llm_aligner=llm_aligner,
                     hp_aligner=hp_aligner)

olala.load(llm_path="upstage/Llama-2-70b-instruct-v2",
           retriever_path="multi-qa-mpnet-base-dot-v1")

alignments = olala.generate(input_data=encoded_ontology)

# 6. OLaLa postprocessing
final_matchings = olala_postprocessor(alignments,
                                      encoded_ontology,
                                      confidence_threshold=0.5,
                                      strict_bad_hosts=False)

# 7. Evaluation
evaluation = metrics.evaluation_report(
    predicts=final_matchings,
    references=dataset["reference"],
)
print("OLaLa Evaluation Report:")
print(json.dumps(evaluation, indent=4))


# 8. XML export
xml_str = xmlify.xml_alignment_generator(matchings=final_matchings)
output_file_path = "olala_matchings.xml"
with open(output_file_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)
print(f"Saved OLaLa matchings to {output_file_path}")

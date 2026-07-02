import json
import torch

from sklearn.linear_model import LogisticRegression

from ontoaligner.ontology import MaterialInformationMatOntoOMDataset, GraphTripleOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.encoder import (
    ConceptParentLightweightEncoder,
    ConceptLLMEncoder,
    ConceptParentRAGEncoder,
    ConceptParentFewShotEncoder,
    GraphTripleEncoder,
)
from ontoaligner.aligner import (
    SimpleFuzzySMLightweight,
    TFIDFRetrieval,
    SBERTRetrieval,
    AutoModelDecoderLLM,
    ConceptLLMDataset,
    MistralLLMBERTRetrieverRAG,
    MistralLLMBERTRetrieverFSRAG,
    TransEAligner,
)
from ontoaligner.postprocess import (
    TFIDFLabelMapper,
    llm_postprocessor,
    graph_postprocessor,
    rag_heuristic_postprocessor,
)
from ontoaligner.aligner.ensemble import EnsembleLearningAligner
from ontoaligner.aligner.ensemble.voting import ReciprocalRankFusionVoting
from ontoaligner import AlignerPipeline

# Step 1: Define paths and runtime settings
source_ontology_path = "assets/MI-MatOnto/mi_ontology.xml"
target_ontology_path = "assets/MI-MatOnto/matonto_ontology.xml"
reference_matching_path = "assets/MI-MatOnto/matchings.xml"

device = "cuda" if torch.cuda.is_available() else "cpu"

ir_model_path = "all-MiniLM-L6-v2"
llm_model_path = "Qwen/Qwen2.5-1.5B-Instruct"


# Step 2: Load datasets
task = MaterialInformationMatOntoOMDataset()
print("Test Task:", task)

dataset = task.collect(
    source_ontology_path=source_ontology_path,
    target_ontology_path=target_ontology_path,
    reference_matching_path=reference_matching_path,
)

graph_dataset = GraphTripleOMDataset().collect(
    source_ontology_path,
    target_ontology_path,
    reference_matching_path,
)


# Step 3: Define LLM mapper
mapper = TFIDFLabelMapper(
    classifier=LogisticRegression(),
    ngram_range=(1, 1),
    label_dict={
        "yes": ["yes", "correct", "true", "same", "equivalent", "valid"],
        "no": ["no", "incorrect", "false", "different", "not same", "invalid"],
    },
)


# Step 4: Define RAG configs
retriever_config = {
    "device": device,
    "top_k": 5,
    "threshold": 0.1,
}

llm_config = {
    "device": device,
    "max_length": 300,
    "max_new_tokens": 10,
    "batch_size": 1,
    "answer_set": {
        "yes": ["yes", "correct", "true", "positive", "valid"],
        "no": ["no", "incorrect", "false", "negative", "invalid"],
    },
}


# Step 5: Define heterogeneous ensemble aligners
aligners = [
    (
        "lightweight",
        AlignerPipeline(
            encoder=ConceptParentLightweightEncoder(),
            aligner=SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2),
            om_dataset=dataset,
        ),
        1.0,
    ),
    (
        "tfidf",
        AlignerPipeline(
            encoder=ConceptParentLightweightEncoder(),
            aligner=TFIDFRetrieval(top_k=5),
            om_dataset=dataset,
            load_params={"path": None},
        ),
        1.0,
    ),
    (
        "sbert",
        AlignerPipeline(
            encoder=ConceptParentLightweightEncoder(),
            aligner=SBERTRetrieval(device=device, top_k=5),
            om_dataset=dataset,
            load_params={"path": ir_model_path},
        ),
        1.0,
    ),
    (
        "kge",
        AlignerPipeline(
            encoder=GraphTripleEncoder(),
            aligner=TransEAligner(
                model="TransE",
                device=device,
                embedding_dim=32,
                num_epochs=1,
                train_batch_size=32,
                eval_batch_size=32,
                num_negs_per_pos=1,
                random_seed=42,
            ),
            om_dataset=graph_dataset,
            postprocessor=graph_postprocessor,
            postprocessor_params={"threshold": 0.0},
        ),
        1.0,
    ),
    (
        "llm",
        AlignerPipeline(
            encoder=ConceptLLMEncoder(),
            aligner=AutoModelDecoderLLM(
                device=device,
                max_length=300,
                max_new_tokens=20,
                batch_size=1,
            ),
            om_dataset=dataset,
            llm_dataset_class=ConceptLLMDataset,
            load_params={"path": llm_model_path},
            postprocessor=llm_postprocessor,
            postprocessor_params={
                "mapper": mapper,
                "interested_class": "yes",
            },
        ),
        1.0,
    ),
    (
        "rag",
        AlignerPipeline(
            encoder=ConceptParentRAGEncoder(),
            aligner=MistralLLMBERTRetrieverRAG(
                retriever_config=retriever_config,
                llm_config=llm_config,
            ),
            om_dataset=dataset,
            load_params={
                "llm_path": llm_model_path,
                "ir_path": ir_model_path,
            },
            postprocessor=rag_heuristic_postprocessor,
            postprocessor_params={
                "topk_confidence_ratio": 3,
                "topk_confidence_score": 3,
            },
        ),
        1.0,
    ),
    (
        "fsrag",
        AlignerPipeline(
            encoder=ConceptParentFewShotEncoder(),
            aligner=MistralLLMBERTRetrieverFSRAG(
                positive_ratio=1.0,
                n_shots=1,
                retriever_config=retriever_config,
                llm_config=llm_config,
            ),
            om_dataset=dataset,
            load_params={
                "llm_path": llm_model_path,
                "ir_path": ir_model_path,
            },
            postprocessor=rag_heuristic_postprocessor,
            postprocessor_params={
                "topk_confidence_ratio": 3,
                "topk_confidence_score": 3,
            },
            include_reference=True,
        ),
        1.0,
    ),
]


# Step 6: Initialize ensemble aligner with rank-based voting
ensemble = EnsembleLearningAligner(
    aligners=aligners,
    voting=ReciprocalRankFusionVoting(k=60),
)


# Step 7: Generate ensemble predictions
matchings = ensemble.generate()

print("\nHeterogeneous Ensemble Matchings")
print("Count:", len(matchings))
print("Top 20:")

for index, item in enumerate(matchings[:20], start=1):
    print(
        f"{index}. "
        f"source={item['source']} | "
        f"target={item['target']} | "
        f"score={item['score']}"
    )


# Step 8: Evaluate ensemble predictions
evaluation = metrics.evaluation_report(
    predicts=matchings,
    references=dataset["reference"],
)

print("\nHeterogeneous Ensemble Evaluation Report:")
print(json.dumps(evaluation, indent=4))


# Step 9: Save XML output
xml_str = xmlify.xml_alignment_generator(matchings=matchings)

with open("ensemble_rrf_all_model_families_matchings.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)

print("Saved XML: ensemble_rrf_all_model_families_matchings.xml")

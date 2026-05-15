import json
from ontoaligner.ontology import FLORAOMDataset
from ontoaligner.encoder import FLORAEncoder
from ontoaligner.aligner.flora import FLORAAligner
from ontoaligner.aligner.flora.postprocessor import  (flora_threshold_postprocessor,
                                                      flora_bilateral_postprocessor,
                                                      flora_1to1_postprocessor)
from ontoaligner.utils import metrics, xmlify

# ---------------------------------------------------------------------------
# Step 1 – Parse: Load Turtle RDF files and extract KG data
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("FLORA Knowledge Graph Alignment Example (OAEI KG Track Usecase).")
print("="*70)
print("\nStep 1: Parsing RDF/Turtle files...")

# FLORAOMDataset.collect() parses both Turtle files and extracts:
#   • Entities (instances)
#   • Predicates (properties)
#   • Triples (RDF facts)
#   • Graph object (for the aligner)
#
# Structure:
#   dataset["source"][0] = {
#       "entities":   [{"iri": "...", "label": "..."}, ...],
#       "predicates": {pred_iri: {count, name, ...}, ...},
#       "triples":    [(subj, pred, obj), ...],
#       "graph":      <FLORA Graph object>     ← used by aligner
#   }

task = FLORAOMDataset(ontology_name = "memoryalpha-stexpanded")

dataset = task.collect(
    source_ontology_path="memoryalpha-stexpanded/source.xml",
    target_ontology_path="memoryalpha-stexpanded/target.xml",
    reference_matching_path="memoryalpha-stexpanded/reference.xml",  # optional: for evaluation
)

source_entities = len(dataset['source'][0]['entities'])
target_entities = len(dataset['target'][0]['entities'])
source_predicates = len(dataset['source'][0]['predicates'])
target_predicates = len(dataset['target'][0]['predicates'])

print(f"  KG1 (Source):  {source_entities} entities, {source_predicates} predicates")
print(f"  KG2 (Target):  {target_entities} entities, {target_predicates} predicates")

# ---------------------------------------------------------------------------
# Step 2 – Encode: Extract Graph objects for the aligner
# ---------------------------------------------------------------------------
print("\nStep 2: Encoding graphs...")

# FLORAEncoder extracts the two FLORA Graph objects from the parsed data.
# These are lightweight representations optimized for the alignment algorithm.
encoder_model = FLORAEncoder()
encoder_output = encoder_model(source=dataset["source"], target=dataset["target"])
# encoder_output == [kg1_graph, kg2_graph]

print(f"  Encoded {len(encoder_output)} graphs")
print(f"  Graph types: {[type(g).__name__ for g in encoder_output]}")

# ---------------------------------------------------------------------------
# Step 3 – Align: Run FLORA with custom parameters
# ---------------------------------------------------------------------------
print("\nStep 3: Aligning knowledge graphs...")
print("  (This may take a few minutes depending on KG size)\n")

# Create aligner with custom parameters
# You can adjust these to tune alignment quality and speed:
aligner = FLORAAligner(
    # Subrelation inference
    alpha=3.0,              # Benefit-of-doubt (3.0=moderate, 1.0=strict, 5.0+=lenient)
    relinit=0.1,            # Initial score for non-identical predicates
    # Literal bootstrapping
    init_threshold=0.2,     # Min semantic similarity (0.0-1.0, higher=stricter)
    string_identity=True,   # Set True to skip embeddings (faster, less accurate)
    # Entity matching
    gramN=100,              # Max evidence facts per entity (increase for more evidence)
    # Convergence
    epsilon=0.01,           # Stop when score change < epsilon
    max_iterations=100,     # Safety cap on iterations
    # Functionality computation
    ngrams=[1, 2],          # N-gram sizes for predicate functionality
    # Optional seed alignments (for supervised variant)
    training_data=None,     # Set to "seed_alignments.tsv" to provide known pairs
    # Embeddings
    # model_id='Lihuchen/pearl_small',  # Hugging Face model for literal similarity,
                                        # set this  if `string_identity` is False
    # device='cuda',         # Uncomment to force GPU (auto-detects by default)
    verbose=True,            # Enable debug logging to see FLORA progress
    workers=30,              # Number of workers for multiprocessing
)

# Run alignment
matchings = aligner.generate(input_data=encoder_output)

print(f"  Found {len(matchings)} entity alignments")
print("\n  Top 5 alignments:")
for i, match in enumerate(matchings[:5], 1):
    print(f"    {i}. {match['source']}")
    print(f"       → {match['target']} (score: {match['score']:.3f})")



# ---------------------------------------------------------------------------
# Step 4 – Do the post-processing!
# ---------------------------------------------------------------------------
print("\nStep 4: Doing the post-processing...")
# Variant 1
instance_alignments, class_alignments, predicate_alignments = flora_threshold_postprocessor(matchings,
                                                                                            prefix1="http://dbkwik.webdatacommons.org/memory-alpha.wikia.com",
                                                                                            prefix2='http://dbkwik.webdatacommons.org/stexpanded.wikia.com',
                                                                                            threshold=0.4)
# Variant 2
bilateral_alignments = flora_bilateral_postprocessor(same_as_scores = aligner.get_same_as_scores(),
                                         source_prefix="http://dbkwik.webdatacommons.org/memory-alpha.wikia.com")

# Variant 3
one2one_alignments = flora_1to1_postprocessor(same_as_scores = aligner.get_same_as_scores())
# ---------------------------------------------------------------------------
# Step 5 – Evaluate: Compare results against reference (if available)
# ---------------------------------------------------------------------------
print("\nStep 5: Evaluating alignment...")
evaluation = metrics.evaluation_report(
    predicts=bilateral_alignments,
    references=dataset["reference"],
)
print("\n  Bilateral  Alignments Evaluation Results:")
print(f"    Precision: {evaluation.get('precision', 'N/A'):.3f}")
print(f"    Recall:    {evaluation.get('recall', 'N/A'):.3f}")
print(f"    F1-Score:  {evaluation.get('f-score', 'N/A'):.3f}")

evaluation = metrics.evaluation_report(
    predicts=one2one_alignments,
    references=dataset["reference"],
)
print("\n  1-to-1  Alignments Evaluation Results:")
print(f"    Precision: {evaluation.get('precision', 'N/A'):.3f}")
print(f"    Recall:    {evaluation.get('recall', 'N/A'):.3f}")
print(f"    F1-Score:  {evaluation.get('f-score', 'N/A'):.3f}")


evaluation = metrics.evaluation_report(
    predicts=instance_alignments,
    references=[item for item in dataset['reference'] if item['type'] == 'instance'],
)
print("\n  Instance alignments Results:")
print(f"    Precision: {evaluation.get('precision', 'N/A'):.3f}")
print(f"    Recall:    {evaluation.get('recall', 'N/A'):.3f}")
print(f"    F1-Score:  {evaluation.get('f-score', 'N/A'):.3f}")

evaluation = metrics.evaluation_report(
    predicts=class_alignments,
    references=[item for item in dataset['reference'] if item['type'] == 'class'],
)
print("\n  Class Alignments Results:")
print(f"    Precision: {evaluation.get('precision', 'N/A'):.3f}")
print(f"    Recall:    {evaluation.get('recall', 'N/A'):.3f}")
print(f"    F1-Score:  {evaluation.get('f-score', 'N/A'):.3f}")

evaluation = metrics.evaluation_report(
    predicts=predicate_alignments,
    references=[item for item in dataset['reference'] if item['type'] == 'predicate'],
)
print("\n  Predicate Alignments Results:")
print(f"    Precision: {evaluation.get('precision', 'N/A'):.3f}")
print(f"    Recall:    {evaluation.get('recall', 'N/A'):.3f}")
print(f"    F1-Score:  {evaluation.get('f-score', 'N/A'):.3f}")

# ---------------------------------------------------------------------------
# Step 6 – Export: Save results to XML and JSON
# ---------------------------------------------------------------------------
print("\nStep 6: Exporting results...")

# Export to XML (OWL alignment format)
xml_str = xmlify.xml_alignment_generator(matchings=matchings)
with open("flora_matchings.xml", "w", encoding="utf-8") as f:
    f.write(xml_str)
print("  ✓ Saved: flora_matchings.xml")

# Export to JSON (convenient for analysis)
with open("flora_matchings.json", "w", encoding="utf-8") as f:
    json.dump(matchings, f, indent=4, ensure_ascii=False)
print("  ✓ Saved: flora_matchings.json")

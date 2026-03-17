"""
FLORA Knowledge Graph Alignment Example
======================================

This example demonstrates the complete FLORA (Fuzzy Logic Knowledge Graph Alignment)
pipeline for unsupervised alignment of two knowledge graphs stored as RDF/Turtle files.

**Algorithm Overview**:

FLORA iteratively aligns entities and relations by:
1. Bootstrapping entity alignments from literal similarity (strings, dates, numbers)
2. Inferring predicate subsumptions from aligned triples
3. Using fuzzy logic rules to align additional entities
4. Repeating until convergence

**Example Pipeline**:

The script follows the OntoAligner standard 5-step workflow:

    1. **Parse**   – Load RDF/Turtle files into FLORA Graphs
    2. **Encode**  – Extract Graph objects for the aligner
    3. **Align**   – Run FLORA with configurable parameters
    4. **Evaluate** – Compare results against reference alignment (if available)
    5. **Export**  – Save results as XML and JSON

**Configuration Options**:

The FLORAAligner class accepts the following key parameters:

    - **alpha** (float): Benefit-of-doubt parameter for subrelation inference.
      Higher values are more lenient. Default: 3.0
    - **init_threshold** (float): Minimum semantic similarity for literal bootstrapping.
      Default: 0.7
    - **gramN** (int): Maximum number of evidential triples per entity.
      Default: 100
    - **epsilon** (float): Convergence threshold for score changes.
      Default: 0.01
    - **max_iterations** (int): Maximum iterations before forced termination.
      Default: 100
    - **string_identity** (bool): Use exact string matching only (no neural embeddings).
      Set True for faster processing on structured data. Default: False
    - **relinit** (float): Initial score for non-identical predicates.
      Default: 0.1
    - **ngrams** (List[int]): N-gram sizes for functionality computation.
      Default: [1, 2]
    - **model_id** (str or None): Hugging Face model ID for semantic embeddings.
      Default: 'Lihuchen/pearl_small' (auto-downloaded if None)
    - **training_data** (str or None): Path to seed alignment file (tab-separated).
      Format: entity1\\tentity2[\\tscore]
    - **device** (str or None): Device for embeddings ('cuda' or 'cpu').
      Auto-detects CUDA if available. Default: None

**Reference**:

    Peng, Yiwen, Bonald, Thomas, and Suchanek, Fabian.
    "FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic."
    In Proc. International Semantic Web Conference (ISWC), 2025.
    https://suchanek.name/work/publications/iswc-2025.pdf

**Common Use Cases**:

Example 1 – Default (Embedding-based literal similarity)::

    aligner = FLORAAligner()
    matchings = aligner.generate(input_data=[kb1_graph, kb2_graph])

Example 2 – Fast mode (String identity only)::

    aligner = FLORAAligner(string_identity=True)
    matchings = aligner.generate(input_data=[kb1_graph, kb2_graph])

Example 3 – Seeded with known alignments::

    aligner = FLORAAligner(training_data="seed_alignments.tsv")
    matchings = aligner.generate(input_data=[kb1_graph, kb2_graph])

Example 4 – Strict matching (low thresholds)::

    aligner = FLORAAligner(
        init_threshold=0.9,     # high similarity required
        alpha=1.0,              # conservative subrelation scoring
        epsilon=0.001           # strict convergence
    )
    matchings = aligner.generate(input_data=[kb1_graph, kb2_graph])
"""

import json
import logging

from ontoaligner.ontology import FLORAOMDataset
from ontoaligner.encoder import FLORAEncoder
from ontoaligner.aligner.flora import FLORAAligner
from ontoaligner.utils import metrics, xmlify

# Enable debug logging to see FLORA progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Step 1 – Parse: Load Turtle RDF files and extract KG data
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("FLORA Knowledge Graph Alignment Example")
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

task = FLORAOMDataset()
task.ontology_name = "My-KG-Pair"

dataset = task.collect(
    source_ontology_path="../assets/cmt-conference/source.xml",
    target_ontology_path="../assets/cmt-conference/target.xml",
    reference_matching_path="../assets/cmt-conference/reference.xml",  # optional: for evaluation
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
    init_threshold=0.7,     # Min semantic similarity (0.0-1.0, higher=stricter)
    string_identity=False,  # Set True to skip embeddings (faster, less accurate)

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
    model_id='Lihuchen/pearl_small',  # Hugging Face model for literal similarity
    # device='cuda',         # Uncomment to force GPU (auto-detects by default)
)

# Run alignment
matchings = aligner.generate(input_data=encoder_output)

print(f"  Found {len(matchings)} entity alignments")
print("\n  Top 5 alignments:")
for i, match in enumerate(matchings[:5], 1):
    print(f"    {i}. {match['source']}")
    print(f"       → {match['target']} (score: {match['score']:.3f})")

# ---------------------------------------------------------------------------
# Step 4 – Evaluate: Compare results against reference (if available)
# ---------------------------------------------------------------------------
if dataset["reference"]:
    print("\nStep 4: Evaluating alignment...")
    evaluation = metrics.evaluation_report(
        predicts=matchings,
        references=dataset["reference"],
    )
    print("\n  Evaluation Results:")
    print(f"    Precision: {evaluation.get('precision', 'N/A'):.3f}")
    print(f"    Recall:    {evaluation.get('recall', 'N/A'):.3f}")
    print(f"    F1-Score:  {evaluation.get('f-score', 'N/A'):.3f}")
    print("\n  Full Report:")
    print(json.dumps(evaluation, indent=4))
else:
    print("\nStep 4: Evaluation skipped (no reference alignment provided)")

# ---------------------------------------------------------------------------
# Step 5 – Export: Save results to XML and JSON
# ---------------------------------------------------------------------------
print("\nStep 5: Exporting results...")

# Export to XML (OWL alignment format)
xml_str = xmlify.xml_alignment_generator(matchings=matchings)
with open("flora_matchings.xml", "w", encoding="utf-8") as f:
    f.write(xml_str)
print("  ✓ Saved: flora_matchings.xml")

# Export to JSON (convenient for analysis)
with open("flora_matchings.json", "w", encoding="utf-8") as f:
    json.dump(matchings, f, indent=4, ensure_ascii=False)
print("  ✓ Saved: flora_matchings.json")

print("\n" + "="*70)
print("Alignment complete!")
print("="*70 + "\n")

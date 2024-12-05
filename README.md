<div align="center">
  🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴Under Construction🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴


  <img src="images/logo-with-background.png" alt="OntoAligner Logo"/>
</div>

<h3 align="center">OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment</h3>

<div align="center">

[![PyPI version](https://badge.fury.io/py/OntoAligner.svg)](https://badge.fury.io/py/OntoAligner)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ontoaligner)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Documentation Status](https://readthedocs.org/projects/ontoaligner/badge/?version=latest)](https://ontoaligner.readthedocs.io/)

</div>

**OntoAligner** is a Python library designed to simplify ontology alignment and matching for researchers, practitioners, and developers. With a modular architecture and robust features, OntoAligner provides powerful tools to bridge ontologies effectively.


## Installation

OntoAligner is available on PyPI and can be installed with pip:

```bash
pip install ontoaligner
```

Alternatively, install the latest version directly from the source:

```bash
git clone git@github.com:sciknoworg/OntoAligner.git
pip install ./ontoaligner
```


## Documentation

Comprehensive documentation for OntoAligner, including detailed guides and examples, is available at **[ontoaligner.readthedocs.io](https://ontoaligner.readthedocs.io/)**.

---

## Quick Tour

Below is an example of using Retrieval-Augmented Generation (RAG) for ontology matching:

```python
import json
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.ontology_matchers import MistralLLMBERTRetrieverRAG
from ontoaligner.encoder import ConceptParentRAGEncoder
from ontoaligner.postprocess import rag_hybrid_postprocessor

# Step 1: Initialize the dataset object for MaterialInformation MatOnto dataset
task = MaterialInformationMatOntoOMDataset()
print("Test Task:", task)

# Step 2: Load source and target ontologies along with reference matchings
dataset = task.collect(
    source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="assets/MI-MatOnto/matchings.xml"
)

# Step 3: Encode the source and target ontologies
encoder_model = ConceptParentRAGEncoder()
encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

# Step 4: Define configuration for retriever and LLM
retriever_config = {
    "device": 'cuda',
    "top_k": 5,
}
llm_config = {
    "device": "cuda",
    "max_length": 300,
    "max_new_tokens": 10,
    "batch_size": 15,
}

# Step 5: Initialize Generate predictions using RAG-based ontology matcher
model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config,
                                   llm_config=llm_config)
predicts = model.generate(input_data=encoded_ontology)

# Step 6: Apply hybrid postprocessing
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(
    predicts=predicts,
    ir_score_threshold=0.1,
    llm_confidence_th=0.8
)

evaluation = metrics.evaluation_report(predicts=hybrid_matchings,
                                       references=dataset['reference'])
print("Hybrid Matching Evaluation Report:", json.dumps(evaluation, indent=4))
print("Hybrid Matching Obtained Configuration:", hybrid_configs)

# Step 7: Convert matchings to XML format and save the XML representation
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)
with open("matchings.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)
```


## Contribution

We welcome contributions to enhance OntoAligner and make it even better! Please review our contribution guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) before getting started. Your support is greatly appreciated.



## Contact

If you encounter any issues or have questions, please submit them in the [GitHub issues tracker](https://github.com/sciknoworg/OntoAligner/issues).


## Citation

If you use OntoAligner in your work or research, please cite the following:

```bibtex
@software{babaei_giglou_ontoaligner_2024,
  author       = {Hamed Babaei Giglou and Jennifer D'Souza and Oliver Karras and S{"o}ren Auer},
  title        = {OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment},
  version      = {1.0.0},
  year         = {2024},
  url          = {https://github.com/HamedBabaei/OntoAligner},
}

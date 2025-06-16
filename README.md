<div align="center">
  <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/main/images/logo-with-background.png" alt="OntoAligner Logo"/>
</div>

<h3 align="center">OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment</h3>

<div align="center">

[![PyPI version](https://badge.fury.io/py/OntoAligner.svg)](https://badge.fury.io/py/OntoAligner)
[![PyPI Downloads](https://static.pepy.tech/badge/ontoaligner)](https://pepy.tech/projects/ontoaligner)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Documentation Status](https://readthedocs.org/projects/ontoaligner/badge/?version=main)](https://ontoaligner.readthedocs.io/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](MAINTANANCE.md)

</div>

**OntoAligner** is a Python library designed to simplify ontology alignment and matching for researchers, practitioners, and developers. With a modular architecture and robust features, OntoAligner provides powerful tools to bridge ontologies effectively.


## 🧪 Installation

You can install **OntoAligner** from PyPI using `pip`:

```bash
pip install ontoaligner
```

Alternatively, to get the latest version directly from the source, use the following commands:

```bash
git clone git@github.com:sciknoworg/OntoAligner.git
pip install ./ontoaligner
```

## 📚 Documentation

Comprehensive documentation for OntoAligner, including detailed guides and examples, is available at **[ontoaligner.readthedocs.io](https://ontoaligner.readthedocs.io/)**. Below are some key tutorials with links to both the documentation and the corresponding example codes.



| Example                                 | Tutorial                                                                                                |                                            Script                                             |
|:----------------------------------------|:--------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------:|
| Lightweight                             | [📚 Fuzzy Matching](https://ontoaligner.readthedocs.io/aligner/lightweight.html)                        |   [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/fuzzy_matching.py)   |
| Retrieval                               | [📚 Retrieval Aligner](https://ontoaligner.readthedocs.io/aligner/retriever.html)                       | [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/retriever_matching.py) |
| Large Language Models                   | [📚 Large Language Models Aligner](https://ontoaligner.readthedocs.io/aligner/llm.html)                 |    [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/llm_matching.py)    |
| Retrieval Augmented Generation          | [📚 Retrieval Augmented Generation](https://ontoaligner.readthedocs.io/aligner/rag.html)                |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/rag_matching.py)|
| FewShot                                 | [📚 FewShot RAG](https://ontoaligner.readthedocs.io/aligner/rag.html#fewshot-rag)                       |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/rag_matching.py)
| In-Context Vectors Learning             | [📚 In-Context Vectors RAG](https://ontoaligner.readthedocs.io/aligner/rag.html#in-context-vectors-rag) |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/icv_rag_matching.py)
| eCommerce  | [📚 Product Alignment in eCommerce](https://ontoaligner.readthedocs.io/usecases/ecommerce.html)                  |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/dev/examples/ecommerce_product_alignment.py)

## 🚀 Quick Tour

Below is an example of using Retrieval-Augmented Generation (RAG) step-by-step approach for ontology matching:

```python
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.aligner import MistralLLMBERTRetrieverRAG
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
retriever_config = {"device": 'cuda', "top_k": 5,}
llm_config = {"device": "cuda", "max_length": 300, "max_new_tokens": 10, "batch_size": 15}

# Step 5: Initialize Generate predictions using RAG-based ontology matcher
model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
model.load(llm_path = "mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")
predicts = model.generate(input_data=encoded_ontology)

# Step 6: Apply hybrid postprocessing
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(predicts=predicts,
                                                            ir_score_threshold=0.1,
                                                            llm_confidence_th=0.8)

evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
print("Hybrid Matching Evaluation Report:", evaluation)

# Step 7: Convert matchings to XML format and save the XML representation
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)
open("matchings.xml", "w", encoding="utf-8").write(xml_str)
```

Ontology alignment pipeline using RAG method:

```python
import ontoaligner

pipeline = ontoaligner.OntoAlignerPipeline(
    task_class=ontoaligner.ontology.MouseHumanOMDataset,
    source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="assets/MI-MatOnto/matchings.xml",
)

matchings, evaluation = pipeline(
    method="rag",
    encoder_model=ontoaligner.encoder.ConceptRAGEncoder(),
    model_class=ontoaligner.aligner.MistralLLMBERTRetrieverRAG,
    postprocessor=ontoaligner.postprocess.rag_hybrid_postprocessor,
    llm_path='mistralai/Mistral-7B-v0.3',
    retriever_path='all-MiniLM-L6-v2',
    llm_threshold=0.5,
    ir_rag_threshold=0.7,
    top_k=5,
    max_length=512,
    max_new_tokens=10,
    device='cuda',
    batch_size=32,
    return_matching=True,
    evaluate=True
)

print("Matching Evaluation Report:", evaluation)
```
## ⭐ Contribution

We welcome contributions to enhance OntoAligner and make it even better! Please review our contribution guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) before getting started. You are also welcome to assist with the ongoing maintenance by referring to [MAINTENANCE.md](MAINTENANCE.md). Your support is greatly appreciated.


If you encounter any issues or have questions, please submit them in the [GitHub issues tracker](https://github.com/sciknoworg/OntoAligner/issues).


## 💡 Acknowledgements

If you use OntoAligner in your work or research, please cite the following preprint:

```bibtex
@inproceedings{babaei2025ontoaligner,
  title={OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment},
  author={Babaei Giglou, Hamed and D’Souza, Jennifer and Karras, Oliver and Auer, S{\"o}ren},
  booktitle={European Semantic Web Conference},
  pages={174--191},
  year={2025},
  organization={Springer}
}
```

This software is archived in Zenodo under the DOI [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14533133.svg)](https://doi.org/10.5281/zenodo.14533133) and is licensed under [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0).

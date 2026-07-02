<div align="center">
  <img src="https://raw.githubusercontent.com/sciknoworg/OntoAligner/main/images/logo-with-background.png" alt="OntoAligner Logo"/>
</div>

<div align="center">

[![PyPI version](https://badge.fury.io/py/OntoAligner.svg)](https://badge.fury.io/py/OntoAligner)
[![PyPI Downloads](https://static.pepy.tech/badge/ontoaligner)](https://pepy.tech/projects/ontoaligner)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Documentation Status](https://readthedocs.org/projects/ontoaligner/badge/?version=main)](https://ontoaligner.readthedocs.io/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](MAINTANANCE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14533133.svg)](https://doi.org/10.5281/zenodo.14533133)

</div>

<h3 align="center">OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment</h3>

**OntoAligner** is a Python library that makes ontology alignment and knowledge graph matching easy for researchers, practitioners, and developers. It ships a single, consistent `parse → encode → align → postprocess` pipeline behind more than a dozen alignment paradigms — from classic fuzzy/lexical matching to retrieval, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Knowledge Graph Embeddings (KGE), fuzzy-logic KG alignment, and ensemble learning — so you can go from two raw ontologies to an evaluated alignment in a handful of lines of code.

> 🏆 OntoAligner was awarded the **[Best Resource Paper Award at ESWC 2025](https://2025.eswc-conferences.org/eswc-2025-best-paper-reviewer-awards/)**.

📘 **New to OntoAligner? Start with the [tutorial notebooks](tutorial/README.md) or the  [full documentation](https://ontoaligner.readthedocs.io/)**.


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

Next, verify the installation:

```python
import ontoaligner

print(ontoaligner.__version__)
```

## 🚀 Quick Tour

### End-to-end with `OntoAlignerPipeline`

The fastest way to run an alignment: pick a dataset, an encoder, and an aligner, and let the pipeline handle collection, encoding, prediction, postprocessing, and evaluation.

```python
import ontoaligner

pipeline = ontoaligner.OntoAlignerPipeline(
    task_class=ontoaligner.ontology.MouseHumanOMDataset,
    source_ontology_path="assets/mouse-human/source.xml",
    target_ontology_path="assets/mouse-human/target.xml",
    reference_matching_path="assets/mouse-human/reference.xml",
)

matchings, evaluation = pipeline(
    method="rag",
    encoder_model=ontoaligner.encoder.ConceptParentRAGEncoder(),
    model_class=ontoaligner.aligner.MistralLLMBERTRetrieverRAG,
    postprocessor=ontoaligner.postprocess.rag_hybrid_postprocessor,
    llm_path="mistralai/Mistral-7B-v0.3",
    retriever_path="all-MiniLM-L6-v2",
    llm_threshold=0.5,
    ir_rag_threshold=0.7,
    top_k=5,
    max_length=512,
    max_new_tokens=10,
    device="cuda",
    batch_size=32,
    return_matching=True,
    evaluate=True,
)

print("Matching Evaluation Report:", evaluation)
```

### Step-by-step, low-level control

Build the same RAG-based alignment yourself for full control over every stage:

```python
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.aligner import MistralLLMBERTRetrieverRAG
from ontoaligner.encoder import ConceptParentRAGEncoder
from ontoaligner.postprocess import rag_hybrid_postprocessor

# Step 1: Initialize the dataset object for the MaterialInformation MatOnto dataset
task = MaterialInformationMatOntoOMDataset()
print("Test Task:", task)

# Step 2: Load source and target ontologies along with reference matchings
dataset = task.collect(
    source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="assets/MI-MatOnto/matchings.xml",
)

# Step 3: Encode the source and target ontologies
encoder_model = ConceptParentRAGEncoder()
encoded_ontology = encoder_model(source=dataset["source"], target=dataset["target"])

# Step 4: Define configuration for retriever and LLM
retriever_config = {"device": "cuda", "top_k": 5}
llm_config = {"device": "cuda", "max_length": 300, "max_new_tokens": 10, "batch_size": 15}

# Step 5: Generate predictions using the RAG-based ontology matcher
model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
model.load(llm_path="mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")
predicts = model.generate(input_data=encoded_ontology)

# Step 6: Apply hybrid postprocessing
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(
    predicts=predicts, ir_score_threshold=0.1, llm_confidence_th=0.8
)

evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset["reference"])
print("Hybrid Matching Evaluation Report:", evaluation)

# Step 7: Convert matchings to XML format and save the XML representation
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)
open("matchings.xml", "w", encoding="utf-8").write(xml_str)
```

### Advanced `AlignerPipeline`
`AlignerPipeline` provides a reusable execution flow for running one user-provided encoder and one ontology matching aligner over a collected ontology matching dataset.  See the bellow on how to define advanced aligner pipeline.

```python
from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
from ontoaligner.utils import metrics
from ontoaligner.encoder import ConceptParentLightweightEncoder
from ontoaligner.aligner import SimpleFuzzySMLightweight
from ontoaligner import AlignerPipeline

task = MaterialInformationMatOntoOMDataset()

dataset = task.collect(
    source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
    target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
    reference_matching_path="assets/MI-MatOnto/matchings.xml",
)

aligner_pipeline = AlignerPipeline(
    encoder=ConceptParentLightweightEncoder(),
    aligner=SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2),
    om_dataset=dataset,
)
matchings = aligner_pipeline.generate()

evaluation = metrics.evaluation_report(predicts=matchings, references=dataset["reference"])
print("Matching Evaluation Report:", evaluation)
```

### Fusing multiple aligners with Ensemble Learning with `AlignerPipeline`

Combine independent aligner branches (lexical, retrieval, KGE, LLM, RAG, ...) into a single, more robust alignment via a voting strategy such as `WeightedVoting`, `BordaVoting`, `CondorcetVoting`, or `ReciprocalRankFusionVoting`:

```python
from ontoaligner.aligner.ensemble import EnsembleLearningAligner
from ontoaligner.aligner.ensemble.voting import WeightedVoting
from ontoaligner import AlignerPipeline

lexical_pipeline = AlignerPipeline(...)  # define your lexical aligner pipeline
retrieval_pipeline = AlignerPipeline(...)  # define your retrieval aligner pipeline
llm_pipeline = AlignerPipeline(...)  # define your LLM aligner pipeline

ensemble = EnsembleLearningAligner(
    aligners=[
        ("lexical", lexical_pipeline, 0.2),   # each branch is an AlignerPipeline
        ("retrieval", retrieval_pipeline, 0.3),
        ("llm", llm_pipeline, 0.5),
    ],
    voting=WeightedVoting(),
)

final_matchings = ensemble.generate()
```

See [ontoaligner.readthedocs.io/developerguide/pipeline.html](https://ontoaligner.readthedocs.io/developerguide/pipeline.html) for more details on how to define your own `AlignerPipeline` and `EnsembleLearningAligner`.

👉 More end-to-end scripts are available in [examples/](examples), including [aligner_pipeline.py](examples/aligner_pipeline.py), [ensemble.py](examples/ensemble.py), [flora.py](examples/flora.py), [olala.py](examples/olala.py), and many more.


## 📚 Documentation & Tutorials

Comprehensive documentation, including detailed guides and examples, is available at **[ontoaligner.readthedocs.io](https://ontoaligner.readthedocs.io/)**. Below are key tutorials with links to both the documentation and the corresponding example scripts.

| Example                        | Tutorial                                                                                                |                                            Script                                             |
|:-------------------------------|:--------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------:|
| Lightweight                    | [📚 Fuzzy Matching](https://ontoaligner.readthedocs.io/aligner/lightweight.html)                        |   [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/fuzzy.py)   |
| Retrieval                      | [📚 Retrieval Aligner](https://ontoaligner.readthedocs.io/aligner/retriever.html)                       | [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/retriever.py) |
| Large Language Models          | [📚 LLM Aligner](https://ontoaligner.readthedocs.io/aligner/llm.html)                                   |    [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/llm.py)    |
| Retrieval Augmented Generation | [📚 RAG Aligner](https://ontoaligner.readthedocs.io/aligner/rag.html)                                   |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/rag.py)|
| FewShot                        | [📚 FewShot-RAG Aligner](https://ontoaligner.readthedocs.io/aligner/rag.html#fewshot-rag)               |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/fewshot_rag.py)|
| In-Context Vectors Learning    | [📚 In-Context Vectors RAG](https://ontoaligner.readthedocs.io/aligner/rag.html#in-context-vectors-rag) |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/icv_rag.py)|
| Knowledge Graph Embedding      | [📚 KGE Aligner](https://ontoaligner.readthedocs.io/aligner/kge.html)                                   |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/kge.py)|
| Property Alignment             | [📚 PropMatch Aligner](https://ontoaligner.readthedocs.io/aligner/propmatch.html)                       |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/property_alignment)|
| FLORA (Knowledge Graphs)       | [📚 FLORA Aligner](https://ontoaligner.readthedocs.io/aligner/flora.html)                                |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/flora.py)|
| OLaLa                          | [📚 OLaLa Aligner](https://ontoaligner.readthedocs.io/aligner/olala.html)                                |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/olala.py)|
| Ensemble Learning              | [📚 Ensemble Learning](https://ontoaligner.readthedocs.io/aligner/ensemble_learning.html)                |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/ensemble.py)|
| eCommerce                      | [📚 Product Alignment in eCommerce](https://ontoaligner.readthedocs.io/usecases/ecommerce.html)          |       [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/usecase/ecommerce_product_alignment.py)|
| Financial Corporate Actions    | [📚 FIBO Corporate Actions Alignment](https://ontoaligner.readthedocs.io/usecases/financial_corporate_actions.html) |  [📝 Code](https://github.com/sciknoworg/OntoAligner/blob/main/examples/usecase/financial_corporate_actions_alignment.ipynb)|

## 👥 Contact & Contributions

We welcome contributions to enhance OntoAligner and make it even better! Please review our contribution guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) before getting started. You are also welcome to assist with the ongoing maintenance by referring to [MAINTENANCE.md](MAINTANANCE.md). Your support is greatly appreciated.

If you encounter any issues or have questions, please submit them in the [GitHub issues tracker](https://github.com/sciknoworg/OntoAligner/issues).


## 📚 Citing this Work

If you use OntoAligner in your work or research, please cite the following preprint:

- OntoAligner Library:
    > Babaei Giglou, H., D'Souza, J., Karras, O., Auer, S. (2025). OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment. In: Curry, E., et al. The Semantic Web. ESWC 2025. Lecture Notes in Computer Science, vol 15719. Springer, Cham. https://doi.org/10.1007/978-3-031-94578-6_10

    📌 BibTeX
    ```bibtex
    @InProceedings{10.1007/978-3-031-94578-6_10,
        author="Babaei Giglou, Hamed and D'Souza, Jennifer and Karras, Oliver and Auer, S{\"o}ren",
        editor="Curry, Edward and Acosta, Maribel and Poveda-Villal{\'o}n, Maria and van Erp, Marieke and Ojo, Adegboyega and Hose, Katja and Shimizu, Cogan and Lisena, Pasquale",
        title="OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment",
        booktitle="The Semantic Web",
        year="2025",
        publisher="Springer Nature Switzerland",
        address="Cham",
        pages="174--191"
    }
    ```

- LLMs4OM (for RAG module)
    >   Babaei Giglou, H., D'Souza, J., Engel, F., Auer, S. (2025). LLMs4OM: Matching Ontologies with Large Language Models. In: Meroño Peñuela, A., et al. The Semantic Web: ESWC 2024 Satellite Events. ESWC 2024. Lecture Notes in Computer Science, vol 15344. Springer, Cham. https://doi.org/10.1007/978-3-031-78952-6_3

    📌 BibTeX
    ```bibtex
    @InProceedings{10.1007/978-3-031-78952-6_3,
      author="Babaei Giglou, Hamed and D'Souza, Jennifer and Engel, Felix and Auer, S{\"o}ren",
      editor="Mero{\~{n}}o Pe{\~{n}}uela, Albert and Corcho, Oscar and Groth, Paul and Simperl, Elena and Tamma, Valentina and Nuzzolese, Andrea Giovanni and Poveda-Villal{\'o}n, Maria and Sabou, Marta and Presutti, Valentina and Celino, Irene and Revenko, Artem and Raad, Joe and Sartini, Bruno and Lisena, Pasquale",
      title="LLMs4OM: Matching Ontologies with Large Language Models",
      booktitle="The Semantic Web: ESWC 2024 Satellite Events",
      year="2025",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="25--35",
      isbn="978-3-031-78952-6"
      }
    ```

- Knowledge Graph Embeddings based aligner

    > Giglou, Hamed Babaei, Jennifer D'Souza, Sören Auer, and Mahsa Sanaei. "OntoAligner Meets Knowledge Graph Embedding Aligners." arXiv preprint arXiv:2509.26417 (2025). https://arxiv.org/abs/2509.26417>

    📌 BibTeX
    ```bibtex
    @article{babaei2025ontoaligner,
      title={OntoAligner Meets Knowledge Graph Embedding Aligners},
      author={Babaei Giglou, Hamed and D'Souza, Jennifer and Auer, S{\"o}ren and Sanaei, Mahsa},
      journal={arXiv e-prints},
      pages={arXiv--2509},
      year={2025}
    }
    ```

## 📃 License

This software is licensed under [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0).

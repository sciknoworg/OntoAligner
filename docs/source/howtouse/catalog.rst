Models Catalog
===============

This catalog provides an organized list of models categorized by type. The table below is structured with the following columns:

- **Category**: The type or category of the aligners.

- **Models with Links**: A list of models belonging to the category.

.. note::
   Use the links to access the implementation details for each model. This catalog helps quickly locate and explore the source code for different models in the **OntoAligner** repository.



.. list-table:: **Model Catalog**
   :header-rows: 1
   :class: catalog-table

   * - **Category**
     - **Models with Links**
   * - **Lightweight Aligners**
     - `SimpleFuzzySMLightweight <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/lightweight/models.py#L23-L47>`__, `WeightedFuzzySMLightweight <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/lightweight/models.py#L50-L74>`__, `TokenSetFuzzySMLightweight <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/lightweight/models.py#L77-L101>`__
   * - **Retrieval Aligners**
     - `AdaRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/retrieval/models.py#L191-L250>`__, `BM25Retrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/retrieval/models.py#L109-L172>`__, `SBERTRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/retrieval/models.py#L28-L42>`__, `SVMBERTRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/retrieval/models.py#L175-L188>`__, `TFIDFRetrieval <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/retrieval/models.py#L45-L106>`__
   * - **LLM Aligners**
     - `AutoModelDecoderLLM <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/llm/models.py#L31-L46>`__, `FlanT5LEncoderDecoderLM <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/llm/models.py#L13-L28>`__, `GPTOpenAILLM <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/llm/models.py#L49-L61>`__
   * - **RAG Aligners**
     - `FalconLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L127-L143>`__, `FalconLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L146-L162>`__, `GPTOpenAILLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L89-L105>`__, `GPTOpenAILLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L108-L124>`__, `LLaMALLMAdaRetrieverRAG <https://ontoaligner.readthedocs.io/package_reference/ontolog_matchers.html#module-ontoaligner.ontology_matchers.rag.models>`__, `LLaMALLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L32-L48>`__, `MPTLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L203-L219>`__, `MPTLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L222-L238>`__, `MambaLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L241-L257>`__, `MambaLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L260-L276>`__, `MistralLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L51-L67>`__, `MistralLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L70-L86>`__, `VicunaLLMAdaRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L165-L181>`__, `VicunaLLMBERTRetrieverRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/rag/models.py#L184-L200>`__
   * - **FewShot-RAG Aligners**
     - `FalconLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L105-L117>`__, `FalconLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L120-L132>`__, `GPTOpenAILLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L75-L87>`__, `GPTOpenAILLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L90-L102>`__, `LLaMALLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L15-L27>`__, `LLaMALLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L30-L42>`__, `MPTLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L165-L177>`__, `MPTLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L180-L192>`__, `MambaLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L195-L207>`__, `MambaLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L210-L222>`__, `MistralLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L45-L57>`__, `MistralLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L60-L72>`__, `VicunaLLMAdaRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L135-L147>`__, `VicunaLLMBERTRetrieverFSRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/fewshot/models.py#L150-L162>`__
   * - **ICV-RAG Aligners**
     - `FalconLLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L53-L69>`__, `FalconLLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L72-L88>`__, `LLaMALLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L15-L31>`__, `LLaMALLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L34-L50>`__, `MPTLLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L129-L145>`__, `MPTLLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L148-L164>`__, `VicunaLLMAdaRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L91-L107>`__, `VicunaLLMBERTRetrieverICVRAG <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/ontology_matchers/icv/models.py#L110-L126>`__


RAG Customization
====================

.. sidebar:: Useful links:

    * `OntoAlignerPipeline Experimentation <https://github.com/sciknoworg/OntoAligner/blob/main/examples/OntoAlignerPipeline-Exp.ipynb>`_


You can use custom LLMs with RAG for alignment. Below, we define two classes, each combining a retrieval mechanism with a LLMs to implement RAG aligner functionality.

.. code-block:: python

    from ontoaligner.ontology_matchers import (
        TFIDFRetrieval,
        SBERTRetrieval,
        AutoModelDecoderRAGLLM,
        AutoModelDecoderRAGLLMV2,
        RAG
    )

    class QwenLLMTFIDFRetrieverRAG(RAG):
        Retrieval = TFIDFRetrieval
        LLM = AutoModelDecoderRAGLLMV2

    class MinistralLLMBERTRetrieverRAG(RAG):
        Retrieval = SBERTRetrieval
        LLM = AutoModelDecoderRAGLLM

As you can see,  **QwenLLMTFIDFRetrieverRAG** Utilizes ``TFIDFRetrieval`` for lightweight retriever with Qwen LLM. While, **MinistralLLMBERTRetrieverRAG** Employs ``SBERTRetrieval`` for retriever using sentence transformers and Ministral LLM.

**AutoModelDecoderRAGLLMV2 and AutoModelDecoderRAGLLM Differences:**

The primary distinction between ``AutoModelDecoderRAGLLMV2`` and ``AutoModelDecoderRAGLLM`` lies in the enhanced functionality of the former. ``AutoModelDecoderRAGLLMV2`` includes additional methods (as presented in the following) for better classification and token validation. Overall, these classes enable seamless integration of retrieval mechanisms with LLM-based generation, making them powerful tools for ontology alignment and other domain-specific applications.


.. code-block:: python

    def get_probas_yes_no(self, outputs):
        """Retrieves the probabilities for the "yes" and "no" labels from model output."""
        probas_yes_no = (outputs.scores[0][:, self.answer_sets_token_id["yes"] +
                                              self.answer_sets_token_id["no"]].float().softmax(-1))
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        """Checks if the tokenizer produces a single token for a given answer string."""
        return len(self.tokenizer(answer).input_ids) == 1

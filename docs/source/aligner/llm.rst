Large Language Models
=============================

Usage
------------

This module guides you through a step-by-step process for performing ontology alignment using a large language model (LLM) and the `OntoAligner` library. By the end, you'll understand how to preprocess data, encode ontologies, generate alignments, evaluate results, and save the outputs in XML and JSON formats.

.. tab:: ➡️ 1: Import

    Start by importing the necessary libraries and modules. These tools will help us process and align the ontologies.

    .. code-block:: python

        # for loading the ontologies, evaluation, and storing the results.
        from ontoaligner.encoder import ConceptLLMEncoder
        from ontoaligner.ontology import MaterialInformationMatOntoOMDataset
        from ontoaligner.utils import metrics, xmlify

        # ontology matchers imports
        from ontoaligner.aligner import AutoModelDecoderLLM, ConceptLLMDataset

        # post-processing imports
        from ontoaligner.postprocess import TFIDFLabelMapper, llm_postprocessor
    ::

.. tab:: ➡️ 2: Parse and Encode

    Define the ontology alignment task using the provided datasets.

    .. code-block:: python

        task = MaterialInformationMatOntoOMDataset()
        print("Test Task:", task)

        dataset = task.collect(
            source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
            target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
            reference_matching_path="assets/MI-MatOnto/matchings.xml"
        )

        encoder_model = ConceptLLMEncoder()
        source_onto, target_onto = encoder_model(source=dataset['source'], target=dataset['target'])

    The encoder module transforms the ontology concepts into a format suitable for the LLM-based matcher. Here the technique used is a concept where it only keeps the concept element from the ``dataset`` for further steps.
    ::

.. tab:: ➡️ 3: Create Dataset

    Prepare the data for the LLM-based matcher by filling in the prompt template.

    .. code-block:: python

        llm_dataset = ConceptLLMDataset(source_onto=source_onto, target_onto=target_onto)


    .. note::
        The dataset formats the encoded concepts into prompts for the LLM to process. Here it is important to have the same type of Encode and Datasets as they operate based on different scenarios. For example, you can not use ``ConceptChildrenLLMEncoder`` with ``ConceptParentLLMDataset``, because ``ConceptChildrenLLMEncoder`` will only keep concept and children for the further step where ``ConceptParentLLMDataset`` only uses concept and parents where due to the missing value here (which is the parent) it will break the pipeline.

    .. tab:: 1️⃣ Sample outputs-1

        The sample output using concept representation (``ConceptLLMEncoder`` and ``ConceptLLMDataset``) of inputs for matching is:

        ::

            [
                {
                  "prompts": "Determine whether the following two concepts refer to the same real-world entity. Respond with 'yes' or 'no' only. \n### Concept 1:\naisi 1000 series steel\n### Concept 2:\nphase equilibrium \n### Your Answer:",
                  "iris": [
                    "http://codata.jp/OML-MaterialInformation#AISI1000SeriesSteel",
                    "http://matonto.org/ontologies/matonto#PhaseEquilibrium"
                  ]
                },
                ...
            ]

        We will proceed with concept only representation!

    .. tab:: 2️⃣ Sample Output-2

        Here is another example sample output using concept-children representation (``ConceptChildrenLLMEncoder`` as encoder and  ``ConceptChildrenLLMDataset`` as LLM dataset):

        ::

            [
                {
                  "prompts": """Determine whether the following two concepts, along with their child categories, refer to the same real-world entity. Respond with 'yes' or 'no' only.\n### Concept 1:\naisi 1000 series steel\n**Children**:\n### Concept 2:\nphase equilibrium\n**Children**:\n### Your Answer: """,
                  "iris": [
                    "http://codata.jp/OML-MaterialInformation#AISI1000SeriesSteel",
                    "http://matonto.org/ontologies/matonto#PhaseEquilibrium"
                  ]
                },
                ...
            ]

    ::


.. tab:: ➡️ 4: Batching

    Use a DataLoader to manage batching. Batching allows the model to process large datasets efficiently in smaller chunks.

    .. code-block:: python

        dataloader = DataLoader(
            llm_dataset,
            batch_size=2048,
            shuffle=False,
            collate_fn=llm_dataset.collate_fn
        )

    ::

.. tab:: ➡️ 5: LLM Aligner

    Set up the LLM-based model for generating alignments.


    .. tab:: ⚙️ Simple setup


        ::

            model = AutoModelDecoderLLM(device='cuda', max_length=300, max_new_tokens=10)
            model.load(path="Qwen/Qwen2-0.5B")


    .. tab:: 🛠️ More complex setup

            You can pick up any parameters that you are willing to proceed.

            ::

                model = AutoModelDecoderLLM(
                                truncation: bool=True,
                                max_length: int=512,
                                max_new_tokens: int=10,
                                padding: bool=True,
                                num_beams: int=1,
                                temperature: float=1.0,
                                top_p: float=1.0,
                                sleep: int=5,
                                device_map: str='balanced'
                       )
                model.load(path="Qwen/Qwen2-0.5B")



    Here we used ``Qwen/Qwen2-0.5B`` model, but feel free to use any LLM you like.

    ::

.. tab:: ➡️ 6: Generate


    Feed batched prompts to the LLM to predict alignments.

    .. code-block:: python

        predictions = []
        for batch in tqdm(dataloader):
            prompts = batch["prompts"]
            sequences = model.generate(prompts)
            predictions.extend(sequences)


    The LLM generates potential alignments between source and target concepts based on the prompts. Here is sample prediction using LLMs.

    .. code-block:: python

        [' \nNo', ' \nNo', ' \nNo',  ' No\n\nConcept 1: Aisi 1',  ' \nYes\nThe Reason is']

    ::

.. tab:: ➡️ 7: Post-Process


    As we see the output of LLM is a text, where it could be hard to determine whether there is a match or not. To ease the process in the Post-Process module we implement multiple label mappers to find the label classes in the output. Here, we refine the predictions using ``TFIDFLabelMapper`` which is based on TF-IDF and logistic regression classifier. The ``llm_postprocessor`` will take predictions and dataset and mapper to find the matchings by only keeping the interested class here (which in a default value is a ``yes`` class).

    .. code-block:: python

        mapper = TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1, 1))
        matchings = llm_postprocessor(predicts=predictions, mapper=mapper, dataset=llm_dataset)


    An important argument for ``TFIDFLabelMapper`` is  ``label_dict`` which the default is set to:

    .. code-block:: javascript

        label_dict = {
            "yes":["yes", "correct", "true"],
            "no":["no", "incorrect", "false"]
        }

    Feel free to change this if you are willing to consider more classes (don't forget to change the prompting in this regard).

    The resulted ``matchings`` will be as following:

    .. code-block:: javascript

        [{'source': 'http://codata.jp/OML-MaterialInformation#AISI5000SeriesSteel',
          'target': 'http://ontology.dumontierlab.com/SecondaryAmineGroup'},
         {'source': 'http://codata.jp/OML-MaterialInformation#AbsorbedDoseRate',
          'target': 'http://ontology.dumontierlab.com/SecondaryAmine'},
         {'source': 'http://codata.jp/OML-MaterialInformation#AbsorbedDoseRate',
          'target': 'http://ontology.dumontierlab.com/SecondaryAmineGroup'},
         {'source': 'http://codata.jp/OML-MaterialInformation#AbsorbedDoseRate',
          'target': 'http://ontology.dumontierlab.com/TertiaryAmineGroup'},
         ... ]
    ::

.. raw:: ➡️ 8: Evaluate and Export


    The following code will compare the generated alignments with reference matchings. Then save the matchings in both XML and JSON formats for further analysis or use. Feel free to use any of the techniques.

    .. code-block:: python

        evaluation = metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
        print("Evaluation Report:", json.dumps(evaluation, indent=4))

    .. tab:: 📄 <> Export matchings to XML

        ::

            xml_str = xmlify.xml_alignment_generator(matchings=matchings)
            with open("matchings.xml", "w", encoding="utf-8") as xml_file:
                xml_file.write(xml_str)

    .. tab:: 🧾 {} Export matchings to JSON

        ::

            with open("matchings.json", "w", encoding="utf-8") as json_file:
                json.dump(matchings, json_file, indent=4, ensure_ascii=False)
    ::

.. hint::

    Prompt are already discussed within this "`LLMs4OM: Matching Ontologies with Large Language Models <https://link.springer.com/chapter/10.1007/978-3-031-78952-6_3>`_" work.

Encoder-Decoder Aligner
-----------------------------

The encoder-decoder aligner leverages models like Flan-T5 to encode input representations and generate ontology alignments in a sequence-to-sequence fashion. This technique is particularly well-suited for encoder-decoder based language models such as BART and Flan-T5, which can effectively capture contextual semantics during both encoding and generation stages. The ``FlanT5LEncoderDecoderLM`` implementation utilizes Google's pre-trained Flan-T5 LLM but is designed with flexibility to support other encoder-decoder architectures.

.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - Encoder-Decoder Aligner
     - Description
     - Link
   * - ``FlanT5LEncoderDecoderLM``
     - This model uses pre-trained LLMs from Google. Nevertheless, this module designed to be used different encoder-decoder based models.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/llm/models.py#L25-L32>`__

To use encoder-decoder based aligner technique:

.. code-block:: python

       from ontoaligner.aligner import FlanT5LEncoderDecoderLM

        model = FlanT5LEncoderDecoderLM()
        model.load(path="...")

Decoder-Only Aligner
-----------------------

The Decoder-Only Aligner provides a wrapper around decoder-only language models—such as GPT2-style models—available via `🤗 HuggingFace <https://huggingface.co/>`_. It enables prompt-based generation for ontology alignment tasks, making it suitable for flexible and autoregressive alignment workflows. The ``AutoModelDecoderLLM`` class supports a wide range of pre-trained LLMs, allowing users to easily load and use models tailored to their needs.


.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - Decoder-Only Aligner
     - Description
     - Link
   * - ``AutoModelDecoderLLM``
     - This model uses pre-trained LLMs. It capable of working with variety of LLMs.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/llm/models.py#L35-L42>`__

To use decoder based aligner based technique:

.. code-block:: python

       from ontoaligner.aligner import FlanT5LEncoderDecoderLM

        model = AutoModelDecoderLLM()
        model.load(path="...")

.. note::

    If the desired model on Hugging Face requires access, please provide your token using the argument: ``huggingface_access_token='...'``.



OpenAI Aligner
--------------------------

The GPT OpenAI Aligner provides an interface to OpenAI’s GPT models (e.g., gpt-4, gpt-3.5-turbo) via the official API, enabling ontology alignment through conversational, prompt-based interaction. This wrapper is designed to streamline alignment workflows using powerful chat-style models, leveraging OpenAI’s instruction-following capabilities for accurate and context-aware alignment generation.


.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - OpenAI GPT Aligner
     - Description
     - Link
   * - ``GPTOpenAILLM``
     - Interfaces with OpenAI’s GPT models via API to generate alignments using conversational prompts.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/llm/models.py#L45-L50>`__


.. code-block:: python

   from ontoaligner.aligner import GPTOpenAILLM

   model = GPTOpenAILLM(openai_api_key="...")
   model.load(path='...') # add desired model id


.. note::

    Make sure to set your OpenAI API key using the ``openai_api_key='...'`` argument when initializing the model.

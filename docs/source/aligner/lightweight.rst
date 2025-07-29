Lightweight
==========================


Usage
-----------------



This module demonstrates the process of aligning ontologies using the **OntoAligner** library. It uses a lightweight fuzzy matching algorithm to match concepts between two ontologies. The example includes data preprocessing, encoding, matching, evaluation, and exporting the matching results in XML or json formats.

.. note::

        * See `Package Reference > Lightweight <../package_reference/aligners.html#module-ontoaligner.aligner.lightweight.lightweight>`_.
        * See `Package Reference > Lightweight Models <../package_reference/aligners.html#module-ontoaligner.aligner.lightweight.models>`_.


.. tab:: ➡️ 1: Import


    Start by importing the necessary libraries and modules. These tools will help us process and align the ontologies.

    .. code-block:: python

        from ontoaligner import ontology, encoder
        from ontoaligner.aligner import SimpleFuzzySMLightweight
        from ontoaligner.utils import metrics, xmlify

    .. tip::

        What does this do:

        - ``ontoaligner``: The main library for ontology alignment.
        - ``SimpleFuzzySMLightweight``: A lightweight matcher for fuzzy alignment that uses simple ratio for matching.
        - ``metrics`` and ``xmlify``: Modules to evaluate and format the results.

    .. note::

        Refer to `Fuzzy Aligners <ightweight.html#fuzzy-aligners>`_ section to read more on Fuzzy Aligners. These models leverage the efficient and robust `rapidfuzz <https://rapidfuzz.github.io/RapidFuzz/index.html>`_ library in the backend, ensuring high-speed and accurate lightweight fuzzy matching for various alignment needs.
    ::


.. tab:: ➡️ 2: Initialize the Task

    Define the ontology alignment task using the provided datasets. This task specifies the source and target ontologies that we will work with.

    .. code-block:: python

        task = ontology.MaterialInformationMatOntoOMDataset()
        print("Test Task:", task)


    This step creates an object that organizes all the required data files and settings for the matching process. The ``print`` statement confirms that the task has been initialized successfully. The ``MaterialInformationMatOntoOMDataset`` class in ``ontology`` endpoint of *OntoAligner* supports source, target, and reference processing of for ``MaterialInformation-MathOnto`` task from `MSE <https://github.com/EngyNasr/MSE-Benchmark>`_ track.

    ::

.. tab:: ➡️ 3: Parse the Dataset


    Next, we load the source ontology, target ontology, and reference matching files. These files are the foundation of our matching process.

    .. code-block:: python

        dataset = task.collect(
            source_ontology_path="../assets/MI-MatOnto/mi_ontology.xml",
            target_ontology_path="../assets/MI-MatOnto/matonto_ontology.xml",
            reference_matching_path="../assets/MI-MatOnto/matchings.xml"
        )

    .. tip::

        What are these files?

        - ``source_ontology_path``: The source ontology path.
        - ``target_ontology_path``: The target ontology path.
        - ``reference_matching_path``: Ground truth data path used for evaluating the results.

    **Outputs Format**: The ``dataset`` is a dictionary with the following key values:

    .. code-block::

        {
            "dataset-info": {
                "track": "mse",
                "ontology-name": "MaterialInformation-MatOnto"
            },
            "source": [
                {
                    "name": "AISI1000SeriesSteel",
                    "iri": "http://codata.jp/OML-MaterialInformation#AISI1000SeriesSteel",
                    "label": "AISI 1000 Series Steel",
                    "childrens": [],
                    "parents": [
                        {
                            "iri": "http://codata.jp/OML-MaterialInformation#FerrousAlloy",
                            "name": "FerrousAlloy",
                            "label": "Ferrous Alloy"
                        }
                    ],
                    "synonyms": [],
                    "comment": []
                }
                ...
            ],
            "target": [
                {
                    "name": "PhaseEquilibrium",
                    "iri": "http://matonto.org/ontologies/matonto#PhaseEquilibrium",
                    "label": "locstr('Phase Equilibrium', 'en')",
                    "childrens": [],
                    "parents": [
                        {
                            "iri": "http://ontology.dumontierlab.com/MeasuredProperty",
                            "name": "MeasuredProperty",
                            "label": "measured property"
                        }
                    ],
                    "synonyms": [],
                    "comment": [
                        "The conditions at which two phases can be at equilibrium"
                    ]
                }
                ...
            ],
            "reference": [
                {
                    "source": "http://codata.jp/OML-MaterialInformation#Density",
                    "target": "http://ontology.dumontierlab.com/Density",
                    "relation": "="
                },
                ...
            ]
        }
    :::

.. tab:: ➡️ 4: Encode


    After loading the dataset, the ``encoder`` module processes and restructures the concepts from the source and target ontologies, preparing them as input for the matching model.

    .. code-block:: python

        encoder_model = encoder.ConceptParentLightweightEncoder()
        encoder_output = encoder_model(
                source=dataset['source'],
                target=dataset['target']
        )


    The ``ConceptParentLightweightEncoder`` utilizes both ``concepts`` and their ``parent`` relationships to reformulate the input representations of ontology concepts, enhancing their comparability. It organizes source and target ontologies for enabling efficient comparison by the fuzzy matching model. The ``encoder_output`` data structure will be as follows:

    .. code-block::

        [
            [
                {
                    "iri": "http://codata.jp/OML-MaterialInformation#AISI1000SeriesSteel",
                    "text": "aisi 1000 series steel  ferrous alloy"
                },
                ...
            ],
            [
                {
                    "iri": "http://matonto.org/ontologies/matonto#PhaseEquilibrium",
                    "text": "phase equilibrium  measured property"
                },
                ...
            ]
        ]
    ::


.. tab:: ➡️ 5: Aligner

    Use the ``SimpleFuzzySMLightweight`` matcher to align concepts by comparing their fuzzy matching scores. The matcher uses a similarity threshold (``0.2`` in this case) to decide which concepts in the source and target ontologies are close enough to be considered a match.

    .. code-block:: python

        model = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.2)
        matchings = model.generate(input_data=encoder_output)


    The ``matchings`` output format will be as follows:

    .. code-block::

        [
            {
                "source": "http://codata.jp/OML-MaterialInformation#AISI1000SeriesSteel",
                "target": "http://matonto.org/ontologies/matonto#PhaseEquilibrium",
                "score": 0.3561643835616438
            },
            ...
        ]
    :::

.. tab:: ➡️ 6: Evaluate


    Evaluate the performance of the fuzzy matcher by comparing the predicted matchings with the reference data.

    .. code-block:: python

        evaluation = metrics.evaluation_report(
            predicts=matchings,
            references=dataset['reference']
        )
        print("Evaluation Report:", json.dumps(evaluation, indent=4))


    A report with metrics like intersection, precision, recall, F1-score, predictions-len, and reference-len which tell you how well the algorithm performed.

    Example output:

    .. code-block::

        {
            "intersection": 40,
            "precision": 7.339449541284404,
            "recall": 13.245033112582782,
            "f-score": 9.445100354191265,
            "predictions-len": 545,
            "reference-len": 302
        }
    ::

.. tab:: ➡️ 7: Export


    Finally, save the matching results in an XML format for future use or integration into other systems.

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

Fuzzy Aligners
------------------------
The ``ontoaligner.aligner.lightweight`` module provides a collection of fuzzy string matching aligners optimized for speed and simplicity. These aligners are ideal for scenarios where fast, lexical-level matching is preferred over complex semantic reasoning.

List of Lightweight Fuzzy Aligners are presented in the following table:

.. list-table::
   :widths: 20 70 10
   :header-rows: 1

   * - Aligner Name
     - Description
     - Link
   * - ``SimpleFuzzySMLightweight``
     - Uses basic string similarity (e.g., Levenshtein, Jaccard) for direct label matching.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/lightweight/models.py#L35-L52>`__
   * - ``WeightedFuzzySMLightweight``
     - Applies weights to label, synonyms, and other features to compute a weighted similarity score.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/lightweight/models.py#L55-L72>`__
   * - ``TokenSetFuzzySMLightweight``
     - Uses token set overlap to align terms with partially matching or rephrased labels.
     - `Source <https://github.com/sciknoworg/OntoAligner/blob/main/ontoaligner/aligner/lightweight/models.py#L75-L92>`__

**Example Usage**

.. code-block:: python

   from ontoaligner.aligner import TokenSetFuzzySMLightweight

   aligner = TokenSetFuzzySMLightweight(threshold=0.8)

   matchings = model.generate(input_data=encoder_output)

.. note::

    - All aligners support a configurable ``threshold`` parameter to control match confidence.
    - These models do not require external embeddings or deep models, making them highly interpretable and lightweight.

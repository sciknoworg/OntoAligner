OntoAligner Documentation
==================================
.. image:: img/logo-ontoaligner.png
   :class: full-width

.. raw:: html

    <div style="text-align: center;">
        <a href="https://badge.fury.io/py/OntoAligner">
            <img src="https://badge.fury.io/py/OntoAligner.svg" alt="PyPI version">
        </a>
        <a href="https://static.pepy.tech/badge/ontoaligner">
            <img src="https://static.pepy.tech/badge/ontoaligner" alt="PyPI downloads">
        </a>
        <a href="https://www.apache.org/licenses/LICENSE-2.0">
            <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
        </a>
        <a href="https://github.com/pre-commit/pre-commit">
            <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="Pre-commit enabled">
        </a>
        <a href="https://ontoaligner.readthedocs.io/en/latest/?badge=main">
            <img src="https://readthedocs.org/projects/ontoaligner/badge/?version=main" alt="Read the Docs">
        </a>
        <a href="https://github.com/sciknoworg/OntoAligner/graphs/commit-activity">
            <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintained">
        </a>
        <a href="https://doi.org/10.5281/zenodo.14533133">
            <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.14533133.svg" alt="DOI">
        </a>
    </div>



Ontologies are a key building block for many applications, such as database integration, knowledge graphs, e-commerce, semantic web services, or social networks. However, evolving systems within the semantic web generally adopt different ontologies. Hence, ontology alignment, the process of identifying correspondences between entities in different ontologies, is a critical task in knowledge engineering. To this endover, **OntoAligner** is a comprehensive modular and robust Python toolkit for ontology alignment built to make ontology alignment/matching easy to use for everyone.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :glob:

   gettingstarted/installation
   gettingstarted/quickstart


.. toctree::
   :maxdepth: 1
   :caption: How to use?

   howtouse/catalog


.. toctree::
   :maxdepth: 1
   :caption: Aligners

   tutorials/lightweight
   tutorials/retriever
   tutorials/llm
   tutorials/rag


.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   package_reference/pipeline
   package_reference/base
   package_reference/ontology
   package_reference/ontolog_matchers
   package_reference/encoder
   package_reference/postprocess
   package_reference/utils

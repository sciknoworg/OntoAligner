OntoAligner Documentation
==================================
.. image:: img/logo-ontoaligner.png
   :class: full-width

.. image:: https://badge.fury.io/py/OntoAligner.svg
    :target: https://badge.fury.io/py/OntoAligner

.. image:: https://img.shields.io/pypi/dm/ontoaligner

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
    :target: https://github.com/pre-commit/pre-commit

.. image:: https://readthedocs.org/projects/ontoaligner/badge/?version=main
    :target: https://ontoaligner.readthedocs.io/en/latest/?badge=main

.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
    :target: https://github.com/sciknoworg/OntoAligner/graphs/commit-activity


Ontologies are a key building block for many applications, such as database integration, knowledge graphs, e-commerce, semantic web services, or social networks. However, evolving systems within the semantic web generally adopt different ontologies. Hence, ontology alignment, the process of identifying correspondences between entities in different ontologies, is a critical task in knowledge engineering. To this endover, **OntoAligner** is a comprehensive modular and robust Python toolkit for ontology alignment built to make ontology alignment/matching easy to use for everyone.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :glob:

   gettingstarted/installation
   gettingstarted/quickstart

.. toctree::
   :maxdepth: 1
   :caption: OntoAligner

   howtouse/usage
   howtouse/ontologymatchers
   howtouse/ontology
   howtouse/encoders


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/lightweight
   tutorials/retriever
   tutorials/llm
   tutorials/rag


.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   package_reference/base
   package_reference/ontology
   package_reference/ontolog_matchers
   package_reference/encoder
   package_reference/postprocess
   package_reference/utils

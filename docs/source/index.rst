
OntoAligner Documentation
===========================

Ontology Alignment (OA) is fundamental for achieving semantic interoperability across diverse knowledge systems. **OntoAligner** (a.k.a Ontology Aligner), a modular Python toolkit for ontology alignment, designed to address current limitations with existing tools faced by practitioners. Existing tools are limited in scalability, modularity, and ease of integration with recent AI advances. OntoAligner provides a flexible architecture integrating existing lightweight OA techniques such as fuzzy matching but goes beyond by supporting contemporary methods with retrieval-augmented generation and large language models for OA. The current framework prioritizes extensibility, enabling researchers to integrate custom alignment algorithms and datasets. With OntoAligner you can handle large-scale ontologies efficiently with few lines of code while delivering high alignment quality. By making OntoAligner open-source, we aim to provide a resource that fosters innovation and collaboration within the OA community, empowering researchers and practitioners with a toolkit for reproducible OA research and real-world applications.

OntoAligner was created by `Scientific Knowledge Organization (SciKnowOrg group) <https://github.com/sciknoworg/>`_ at `Technische Informationsbibliothek (TIB) <https://www.tib.eu/de/>`_. Don't hesitate to open an issue on the `OntoAligner repository <https://github.com/sciknoworg/OntoAligner>`_ if something is broken or if you have further questions.


.. note::

    OntoAligner was  awarded the `🏆 Best Resource Paper Award at ESWC 2025 <https://2025.eswc-conferences.org/eswc-2025-best-paper-reviewer-awards/>`_

.. raw:: html

    <div class="project-vision">
      <strong>The vision is to create a unified hub that brings together a wide range of ontology alignment models, making integration seamless for researchers and practitioners.</strong>
    </div>

.. raw:: html

	<div class="video-card">
	  <iframe
	    src="https://videolectures.net/embed/videos/eswc2025_bernardin_babaei_giglu?part=1"
	    frameborder="0"
	    allowfullscreen>
	  </iframe>
	  <p class="video-caption">
	    ESWC 2025 Talk — OntoAligner Presentation by Hamed Babaei Giglou.
	  </p>
	</div>


Citing
=========

If you find this repository helpful, feel free to cite our publication `OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment <https://link.springer.com/chapter/10.1007/978-3-031-94578-6_10>`_:

 .. code-block:: bibtex

    @inproceedings{babaei2025ontoaligner,
      title={OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment},
      author={Babaei Giglou, Hamed and D’Souza, Jennifer and Karras, Oliver and Auer, S{\"o}ren},
      booktitle={European Semantic Web Conference},
      pages={174--191},
      year={2025},
      organization={Springer}
    }

or our related work `LLMs4OM: Matching Ontologies with Large Language Models <https://link.springer.com/chapter/10.1007/978-3-031-78952-6_3>`_:

 .. code-block:: bibtex

  @inproceedings{babaei2024llms4om,
      title={LLMs4OM: Matching Ontologies with Large Language Models},
      author={Babaei Giglou, Hamed and D’Souza, Jennifer and Engel, Felix and Auer, S{\"o}ren},
      booktitle={European Semantic Web Conference},
      pages={25--35},
      year={2024},
      organization={Springer}
    }

or if you are using Knowledge Graph Embeddings refer to `OntoAligner Meets Knowledge Graph Embedding Aligners <https://arxiv.org/abs/2509.26417>`_:

 .. code-block:: bibtex

  @article{babaei2025ontoaligner,
	  title={OntoAligner Meets Knowledge Graph Embedding Aligners},
	  author={Babaei Giglou, Hamed and D'Souza, Jennifer and Auer, S{\"o}ren and Sanaei, Mahsa},
	  journal={arXiv e-prints},
	  pages={arXiv--2509},
	  year={2025}
	}



.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :glob:
   :hidden:

   gettingstarted/overview
   gettingstarted/installation
   gettingstarted/quickstart

.. toctree::
   :caption: Developer Guide
   :titlesonly:
   :hidden:
   :maxdepth: 1

   developerguide/parsers

.. toctree::
   :caption: Aligners
   :titlesonly:
   :hidden:
   :maxdepth: 1

   aligner/lightweight
   aligner/retriever
   aligner/llm
   aligner/rag
   aligner/kge
   aligner/propmatch

.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :hidden:

   usecases/ecommerce


.. toctree::
   :maxdepth: 1
   :caption: Package Reference
   :hidden:

   package_reference/pipeline
   package_reference/parsers
   package_reference/encoders
   package_reference/aligners
   package_reference/postprocess

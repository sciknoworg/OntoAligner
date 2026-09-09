SSSOM
=====================================================

.. sidebar:: Useful links:

    * `A Simple Standard for Sharing Ontology Mappings (SSSOM) <https://mapping-commons.github.io/sssom/dev/>`_
    * `OntoAligner SSSOM Utility <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/utils/sssom.py>`_
    * `OntoAligner SSSOM-Pydantic Utility <https://github.com/sciknoworg/OntoAligner/blob/dev/ontoaligner/utils/sssom_pydantic.py>`_


The Simple Standard for Sharing Ontological Mappings (SSSOM) is a community-driven
standard for representing and exchanging semantic mappings in a consistent format.
It provides a common structure for sharing ontology mappings across different tools,
datasets, and systems, making mappings easier to exchange, integrate, and reuse.

The SSSOM data model is centered around two main concepts: **mappings** and
**mapping sets**. A mapping represents a correspondence between two semantic
entities as a ``subject``-``predicate``-``object`` statement. In addition to this
core correspondence, SSSOM can describe metadata such as mapping justification,
confidence, provenance, and other information about how the mapping was produced.

A mapping set represents a collection of mappings and can contain additional
metadata describing the collection, such as its identifier, license, or description.
The mappings are commonly exchanged using the SSSOM TSV format.

OntoAligner provides two utilities for converting generated ontology alignments
to SSSOM TSV format:

* ``ontoaligner.utils.sssom`` uses the `sssom-py <https://github.com/mapping-commons/sssom-py>`_ toolkit.
* ``ontoaligner.utils.sssom_pydantic`` uses the `sssom-pydantic <https://github.com/cthoyt/sssom-pydantic/tree/main>`_ implementation.


Usage
----------------------------

.. tab:: 📄 sssom

    The ``ontoaligner.utils.sssom`` module converts OntoAligner matchings into
    SSSOM TSV using ``sssom-py``.

    This implementation can use OntoAligner aligner and postprocessor information
    to derive suitable mapping justification and score metadata when the score
    semantics are known.

    The generated SSSOM output can include:

    .. list-table::
       :header-rows: 1
       :widths: 32 68

       * - Field
         - Description
       * - ``subject_id``
         - Source ontology entity represented as a CURIE using the configured converter.
       * - ``predicate_id``
         - Mapping predicate supplied through ``predicate_id``.
       * - ``object_id``
         - Target ontology entity represented as a CURIE using the configured converter.
       * - ``mapping_justification``
         - Explicit or automatically derived SemAPV mapping justification.
       * - ``subject_label``
         - Source ontology label when available from the parser output.
       * - ``object_label``
         - Target ontology label when available from the parser output.
       * - ``similarity_score``
         - Similarity score when the aligner score semantics are supported.
       * - ``similarity_measure``
         - Description of the similarity function associated with the score.
       * - ``confidence``
         - Confidence value for supported confidence-based workflows.
       * - MappingSet metadata
         - Metadata such as ``mapping_set_id``, ``license``, ``mapping_tool``,
           ``mapping_tool_version``, and other supplied MappingSet fields.

    The general flow uses final OntoAligner ``source``-``target`` matchings and
    passes the alignment context to ``sssom_alignment_generator``. A complete SSSOM usage example is available at `examples/retriever_sssom.py <https://github.com/sciknoworg/OntoAligner/blob/dev/examples/retriever_sssom.py>`_.

    .. code-block:: python

        from ontoaligner.utils.sssom import sssom_alignment_generator

        sssom_str = sssom_alignment_generator(
            matchings=matchings,
            source=dataset["source"],
            target=dataset["target"],
            predicate_id="owl:equivalentClass",
            mapping_set_metadata={
                "mapping_set_id": "https://example.org/mappings/mouse-human-sbert",
                "license": "https://creativecommons.org/licenses/by/4.0/",
                "subject_type": "owl:Class",
                "object_type": "owl:Class",
            },
            curie_map={
                "mouse": "http://mouse.owl#",
                "human": "http://human.owl#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "semapv": "https://w3id.org/semapv/vocab/",
            },
            aligner=model,
            postprocessor=retriever_postprocessor,
            postprocessor_params={
                "threshold": threshold,
            },
        )

    SSSOM output can also be generated from mappings produced with
    ``AlignerPipeline``.

    .. code-block:: python

        from ontoaligner.utils.sssom import sssom_alignment_generator

        sssom_str = sssom_alignment_generator(
            matchings=matchings,
            pipeline=aligner_pipeline,
            predicate_id="owl:equivalentClass",
            mapping_set_metadata={
                "mapping_set_id": "https://example.org/mappings/mouse-human-sbert",
                "license": "https://creativecommons.org/licenses/by/4.0/",
                "subject_type": "owl:Class",
                "object_type": "owl:Class",
            },
            curie_map={
                "mouse": "http://mouse.owl#",
                "human": "http://human.owl#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "semapv": "https://w3id.org/semapv/vocab/",
            },
        )

    When ``pipeline`` is provided, the exporter uses the pipeline aligner or
    reranker, postprocessor, postprocessor parameters, and available source
    and target ontology entities.

    .. note::

        CURIE handling follows these rules:

        * Provide ``curie_map`` explicitly for custom or unregistered namespaces.
        * An explicit ``curie_map`` should contain all prefixes used by mappings and metadata.
        * If ``curie_map`` is omitted, Bioregistry is used as a fallback for registered namespaces.


.. tab:: 🐍 sssom-pydantic

    The ``ontoaligner.utils.sssom_pydantic`` module provides an alternative
    SSSOM export path based on ``sssom-pydantic``, an opinionated Pydantic-based
    implementation of a subset of SSSOM.

    OntoAligner matchings are converted into typed ``SemanticMapping`` objects
    before serialization. The resulting objects can also be accessed directly
    through ``to_semantic_mappings`` for further inspection or processing.

    The current OntoAligner implementation can populate:

    .. list-table::
       :header-rows: 1
       :widths: 32 68

       * - Field
         - Description
       * - ``subject``
         - Parsed source ontology entity reference.
       * - ``predicate``
         - Typed mapping predicate derived from the matching ``relation`` when supported.
       * - ``object``
         - Parsed target ontology entity reference.
       * - ``justification``
         - Mapping justification represented as an SSSOM reference.
       * - ``confidence``
         - OntoAligner ``score`` represented as a confidence value when it falls within the valid ``0`` to ``1`` range.
       * - ``mapping_date``
         - Date associated with the generated mapping.
       * - MappingSet metadata
         - Optional ``sssom_pydantic.MappingSet`` metadata supplied to the writer.

    Generate SSSOM TSV output directly from OntoAligner matchings using
    ``sssom_alignment_generator``. A complete SSSOM-Pydantic usage example is available at `examples/retriever_sssom_pydantic.py <https://github.com/sciknoworg/OntoAligner/blob/dev/examples/retriever_sssom_pydantic.py>`_.

    .. code-block:: python

        import curies
            import sssom_pydantic as spd

            from ontoaligner.utils.sssom_pydantic import sssom_alignment_generator

            converter = curies.Converter.from_prefix_map(
                {
                    "mouse": "http://mouse.owl#",
                    "human": "http://human.owl#",
                }
            )

            metadata = spd.MappingSet(
                id="https://example.org/mappings/mouse-human",
            )

            sssom_alignment_generator(
                matchings=matchings,
                path="mouse-human.sssom.tsv",
                converter=converter,
                metadata=metadata,
            )


    Typed ``SemanticMapping`` objects can also be generated without immediately
    writing the SSSOM file.

    .. code-block:: python

        from ontoaligner.utils.sssom_pydantic import to_semantic_mappings

            semantic_mappings = to_semantic_mappings(
                matchings=matchings,
                converter=converter,
            )

            print(semantic_mappings[0])

    .. note::

        CURIE handling follows these rules:

        * A custom ``curies.Converter`` can be supplied explicitly.
        * If no converter is supplied, Bioregistry is used as the default converter.
        * Mappings whose source or target cannot be converted to a CURIE are skipped and logged.



Configuration
--------------------

.. tab:: 🔧 sssom

    ``sssom_alignment_generator`` provides control over SSSOM mapping metadata,
    CURIE handling, and OntoAligner alignment context.

    .. list-table::
       :header-rows: 1
       :widths: 24 18 16 42

       * - Parameter
         - Type
         - Default
         - Description
       * - **matchings**
         - List[Dict]
         - —
         - OntoAligner matchings containing ``source``, ``target``, and optionally score information.
       * - **source**
         - List[Dict]
         - ``None``
         - Parsed source ontology entities used to populate ``subject_label``.
       * - **target**
         - List[Dict]
         - ``None``
         - Parsed target ontology entities used to populate ``object_label``.
       * - **predicate_id**
         - str
         - —
         - SSSOM predicate describing the relation between mapped entities.
       * - **mapping_set_metadata**
         - dict
         - —
         - MappingSet information including the required ``mapping_set_id`` and ``license``.
       * - **curie_map**
         - dict
         - ``None``
         - Prefix-to-IRI mappings used for CURIE conversion.
       * - **pipeline**
         - AlignerPipeline
         - ``None``
         - Pipeline used to obtain aligner, reranker, postprocessor, parameters, and ontology entities.
       * - **aligner**
         - BaseOMModel
         - ``None``
         - Aligner used when mappings are generated without ``AlignerPipeline``.
       * - **postprocessor**
         - Any
         - ``None``
         - Postprocessor applied to aligner predictions.
       * - **postprocessor_params**
         - dict
         - ``None``
         - Parameters supplied to the postprocessor.
       * - **mapping_justification**
         - str
         - ``None``
         - Explicit mapping justification overriding automatic inference.
       * - **include_aligner_metadata**
         - bool
         - ``True``
         - Whether OntoAligner derives mapping justification and supported score metadata from the alignment workflow.

    .. note::

        ``mapping_set_metadata`` must contain ``mapping_set_id`` and ``license``.
        Additional valid SSSOM MappingSet metadata can also be supplied in the same
        dictionary.

.. tab:: 🔧 sssom-pydantic

    ``sssom_alignment_generator`` converts OntoAligner matchings into
    ``SemanticMapping`` objects and writes the generated mappings using
    ``sssom_pydantic.write``.

    .. list-table::
       :header-rows: 1
       :widths: 24 18 16 42

       * - Parameter
         - Type
         - Default
         - Description
       * - **matchings**
         - Iterable[Mapping]
         - —
         - OntoAligner dict-like matchings containing at least ``source`` and ``target``.
       * - **path**
         - str, Path, TextIO
         - —
         - Output file path or open file-like object used to write the SSSOM TSV.
       * - **converter**
         - curies.Converter
         - ``None``
         - Converter used to convert source and target IRIs into CURIE references.
       * - **metadata**
         - MappingSet
         - ``None``
         - Optional ``sssom_pydantic.MappingSet`` metadata supplied to the writer.
       * - ****kwargs**
         - Any
         - ``{}``
         - Additional keyword arguments forwarded to ``sssom_pydantic.write``.

    .. hint::

        ``to_semantic_mappings`` can be used when typed mapping objects are needed
        without immediately serializing the output.

    .. tab:: 🔧 ``to_semantic_mappings``

        .. list-table::
           :header-rows: 1
           :widths: 24 18 16 42

           * - Parameter
             - Type
             - Default
             - Description
           * - **matchings**
             - Iterable[Mapping]
             - —
             - OntoAligner mappings to convert to ``SemanticMapping`` objects.
           * - **converter**
             - curies.Converter
             - ``None``
             - Converter used to convert source and target ontology identifiers.

    .. note::

        If ``converter`` is not supplied, the Bioregistry default converter is used.
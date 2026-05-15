
Metrics
========

.. sidebar:: **Alignment Format**

   Every function expects alignments as lists of dictionaries with at least ``source`` and ``target`` keys.
   Predictions may additionally carry a ``score`` (float) used by the ranking metrics.

   .. code-block:: python

      # Alignment entry format
      predict  = {
                "source": "owl:Class_A",
                "target": "owl:Class_X",
                "score": 0.92
      }
      reference = {
                "source": "owl:Class_A",
                "target": "owl:Class_X",
                "relation": "="
      }



The ``ontoaligner.utils.metrics`` module provides evaluation functions for ontology alignment tasks.
Given a list of **predicted** alignments and a list of **reference** (ground-truth) alignments,
it computes standard IR-style metrics as well as ranking-based measures.


Standard Metrics
-----------------

These metrics treat alignments as sets of ``(source, target)`` pairs.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Function
     - Description
   * - ``calculate_intersection``
     - Count of unique ``(source, target)`` pairs present in both predictions and references.
   * - ``precision_score``
     - Fraction of predictions that are correct.
   * - ``recall_score``
     - Fraction of reference alignments that were retrieved.
   * - ``f1_measurement``
     - Weighted harmonic mean of precision and recall (Fβ).
   * - ``evaluation_report``
     - Returns all of the above in a single summary dictionary (values scaled to 0–100).


----

.. rubric:: Precision

.. math::

   P = \frac{|\,\text{predicts} \cap \text{references}\,|}{|\,\text{predicts}\,|}

Returns ``0`` when the prediction list is empty.


----

.. rubric:: **Recall**

.. math::

   R = \frac{|\,\text{predicts} \cap \text{references}\,|}{|\,\text{references}\,|}

Returns ``0`` when the reference list is empty.


----

.. rubric:: **Fβ-Score**

.. math::

   F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}

Set ``beta=1`` (default) for the standard F1-measurement, ``beta=2`` to weight recall more heavily,
or ``beta=0.5`` to favour precision.


----

.. rubric:: ``evaluation_report`` — output keys

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``intersection``
     - int
     - Number of matched alignment pairs
   * - ``precision``
     - float
     - Precision × 100
   * - ``recall``
     - float
     - Recall × 100
   * - ``f-score``
     - float
     - Fβ × 100
   * - ``predictions-len``
     - int
     - Total number of predictions supplied
   * - ``reference-len``
     - int
     - Total number of reference alignments




Ranking Metrics
---------------------

These metrics require a ``score`` field on each prediction and evaluate the **ranking quality**
of the predictions for each source concept.


----

.. rubric:: **Hit@K**

.. math::

   \text{Hit@K} = \frac{1}{|\text{references}|} \sum_{\text{ref}} \mathbf{1}\!\left[\text{ref.target} \in \text{top-}K(\text{ref.source})\right]

The fraction of reference alignments whose correct target appears in the top-*K* predictions
for the same source (ranked by descending score). Returns ``0`` if ``k ≤ 0`` or no references exist.


----

.. rubric:: **Mean Reciprocal Rank (MRR)**

.. math::

   \text{MRR} = \frac{1}{|\text{references}|} \sum_{\text{ref}} \frac{1}{\text{rank}(\text{ref.target})}

The average reciprocal rank of the correct target in the prediction list for each source.
Missing predictions contribute ``0`` to the sum.



Example Usage
-------------

.. code-block:: python

   from ontoaligner.utils.metrics import evaluation_report, hit_at_k, mrr

   predicts = [
       {"source": "A", "target": "1", "score": 0.9},
       {"source": "A", "target": "2", "score": 0.5},
       {"source": "B", "target": "3", "score": 0.8},
   ]
   references = [
       {"source": "A", "target": "1"},
       {"source": "B", "target": "3"},
   ]

   report = evaluation_report(predicts, references)

   print(f"Hit@1 : {hit_at_k(predicts, references, k=1):.2f}")
   print(f"Hit@2 : {hit_at_k(predicts, references, k=2):.2f}")
   print(f"MRR   : {mrr(predicts, references):.2f}")

.. tip::

   Use ``evaluation_report`` for a quick benchmark summary; add ``hit_at_k`` and ``mrr``
   when your matcher produces ranked candidate lists (e.g., retrieval-based or LLM-based aligners).

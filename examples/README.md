# 📚 OntoAligner Examples

Practical examples demonstrating various ontology alignment and matching techniques using OntoAligner.

**OntoAligner Pattern:**

```python
# 1. Load dataset
task = MaterialInformationMatOntoOMDataset()
dataset = task.collect(source_ontology_path="...", target_ontology_path="...", reference_matching_path="...")

# 2. Encode
encoder = ConceptParentLightweightEncoder()
encoded = encoder(source=dataset['source'], target=dataset['target'])

# 3. Align
aligner = SomeAligner()
matchings = aligner.generate(input_data=encoded)

# 4. Evaluate
metrics.evaluation_report(predicts=matchings, references=dataset['reference'])
```

Run any example:
```bash
python fuzzy_matching.py
```

**📖 Related Resources:**

- **Documentation**: https://ontoaligner.readthedocs.io
- **PyPI**: https://pypi.org/project/OntoAligner/
- **Tutorial**: See [tutorial/](../tutorial/) directory
- **GitHub**: https://github.com/sciknoworg/OntoAligner

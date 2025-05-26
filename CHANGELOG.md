## Changelog

### V1.4.1 Changelog (May 26, 2025)
- Fixed an issue related to #25 and #36.
- Sentence-transformer v4.1.0 is supported.
- Adding Python 3.12 and 3.13 for automated testing.
- Remove the dependency with ontospy since it is not being maintained. Initially it was used for `MaterialInformationOntoOntology` class.

### V1.4.0 Changelog (May 22, 2025)
- Fixed a security vulnerability by updating the Torch and Transformers dependency version.
- Integrated pytest into the pyproject.toml to enable testing support.
- Resolved Python version compatibility issues in the continuous integration (CI) pipeline for stable test runs.
- Included a preprint citation to enhance academic referencing.
- Made cosmetic and informational updates to the documentation and readme to improve readability.
- Renamed the module ontology_matchers to aligner for better clrity and consistency.
- Added ecommerce use-case examples and documentation to demonstrate real-world applications.

### V1.3.0 (March 21, 2025)
- Added a GenericOntology and GenericOMDataset.
- Added a documentation for OA tasks.
- Added the test for GenericOntology.
- Added the bug fix.


### V1.2.3 (Feb 4, 2025)
- Minor fix to the CITATION.md, CONTRIBUTING.md, README.md, and documentations.
- Added MAINTANENCE.md for maintanence plan.
- Added CODE_OF_CONDUCT.md principles for community engagements and collaborations.
- Added CHANGELOG.md
- Added new version.
- Fix bugs in retriever, pipeline, and rag modules.
- Added new dependencies and update version of `setfit` library.

### v1.2.2 (Dec 20, 2024)
- Minor fix
- License update

### v1.2.1 (Dec 13, 2024)
- Documentation updates
- Fixed Hugging Face requirements
- Documentation corrections

### v1.2.0 (Dec 10, 2024)
- Bug fixes in RAG dataset and modules
- Added OntoAlignerPipeline
- Documentation updates

### v1.1.0 (Dec 8, 2024)
- Added Fewshot RAG and ICV RAG
- Fixed imports for Fewshot and ICV RAG
- Completed examples and initial pipeline added

### v1.0.0 (Dec 6, 2024)
- Major updates on ontology classes, base classes, and encoders
- Extended post-processing with label mappers
- Documentation overhaul

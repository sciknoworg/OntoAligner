from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="OntoAligner",
    version="1.2.2",
    author="Hamed Babaei Giglou",
    author_email="hamedbabaeigiglou@gmail.com",
    description="OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sciknoworg/OntoAligner",
    packages=find_packages(),
    install_requires=[
        "pathlib",
        "argparse",
        "datasets",
        "numpy",
        "pandas",
        "scikit-learn",
        "ontospy==2.1.1",
        "openai==1.56.0",
        "owlready2==0.44",
        "rank_bm25==0.2.2",
        "rapidfuzz==3.5.2",
        "rdflib==7.1.1",
        "sentence-transformers==2.2.2",
        "setfit==1.1.1",
        "torch==2.5.0",
        "tqdm==4.66.3",
        "transformers==4.46.0",
        "huggingface_hub==0.23.5",
        "bitsandbytes==0.45.1"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<4.0.0",
    project_urls={
        "Documentation": "https://ontoaligner.readthedocs.io/",
        "Source": "https://github.com/sciknoworg/OntoAligner",
        "Tracker": "https://github.com/sciknoworg/OntoAligner/issues",
    },
)

from setuptools import setup, find_packages
import os

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="OntoAligner",
    version=open(os.path.join(os.path.dirname(__file__), 'ontoaligner/VERSION')).read().strip(),
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
        "tqdm",
        "openai==1.56.0",
        "owlready2==0.44",
        "rank_bm25==0.2.2",
        "rapidfuzz==3.5.2",
        "rdflib==7.1.1",
        "sentence-transformers>=5.1.0,<6.0.0",
        "torch>=2.8.0,<3.0.0",
        "transformers>=4.56.0,<5.0.0",
        "huggingface-hub>=0.34.4,<1.0.0",
        "bitsandbytes>=0.45.1,<1.0.0; platform_system == 'Linux'",
        "pykeen==1.11.1"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<4.0.0",
    project_urls={
        "Documentation": "https://ontoaligner.readthedocs.io/",
        "Source": "https://github.com/sciknoworg/OntoAligner",
        "Tracker": "https://github.com/sciknoworg/OntoAligner/issues",
    },
)

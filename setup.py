from setuptools import setup, find_packages

setup(
    name="OntoAligner",
    version="0.1.1",
    author="Hamed Babaei Giglou",
    author_email="hamedbabaeigiglou@gmail.com",
    description="OntoAligner: A Ontology Alignment Python Library",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HamedBabaei/OntoAligner",
    packages=find_packages(),
    install_requires=[
        "pathlib",
        "argparse",
        "datasets==2.19.1",
        "numpy",
        "pandas",
        "ontospy==2.1.1",
        "openai==1.56.0",
        "owlready2==0.44",
        "rank_bm25==0.2.2",
        "rapidfuzz==3.5.2",
        "rdflib==7.1.1",
        "scikit_learn==1.3.2",
        "sentence_transformers==2.2.2",
        "setfit==1.1.0",
        "torch==2.5.0",
        "tqdm==4.66.3",
        "transformers==4.46.0"
    ],
    classifiers=[  # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)

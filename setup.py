from setuptools import setup, find_packages

setup(
    name="OntoAligner",
    version="0.1.0",
    author="Hamed Babaei Giglou",
    author_email="hamedbabaeigiglou@gmail.com",
    description="A short description of your library",
    long_description=open('README.md').read(),  # Optional: Provide a long description using README.md
    long_description_content_type="text/markdown",
    url="https://github.com/HamedBabaei/OntoAligner",
    packages=find_packages(),
    install_requires=[
        "pathlib",
        "argparse",
        "owlready2",
        "rdflib",
        "tqdm",
        "ontospy",
        "numpy",
        "torch",
        "contextlib",
        "transformers",
        "datasets",
        "rapidfuzz",
        "openai",
        "rank_bm25",
        "scikit-learn",
        "sentence-transformers",
        "pre-commit"
    ],
    classifiers=[  # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

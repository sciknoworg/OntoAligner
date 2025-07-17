Installation
=============

We recommend **Python 3.10+**, `PyTorch 1.4.0+ <https://pytorch.org/get-started/locally/>`_, and `transformers v4.41.0+ <https://github.com/huggingface/transformers>`_.


Install with pip
-----------------------

.. sidebar:: Verify the installation

    Once the isntallation is done, verify the installation by:

    .. code-block:: python

        import ontoaligner

        print(ontoaligner.__version__)


.. tab:: From PyPI

    OntoAligner is available on the Python Package Index at `pypi.org <https://pypi.org/project/OntoAligner/>`_ for installation.
    ::

        pip install -U OntoAligner

.. tab:: From GitHub

    The following pip install will installs the latest version of OntoAligner from the `main` branch of the OntoAligner at GitHub using `pip`.

    ::

        pip install git+https://github.com/sciknoworg/OntoAligner.git


Install from Source
----------------------
You can install OntoAligner directly from source to take advantage of the bleeding edge main branch for development.


1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/sciknoworg/OntoAligner.git
    cd OntoAligner

2. (Optional but recommended) Create and activate a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies and the library

.. code-block:: bash

    pip install -e .

.. hint:: The -e flag installs the package in editable mode, which is ideal for development—changes in the code reflect immediately.

Install PyTorch with CUDA support
--------------------------------------------
To use a GPU/CUDA for aligner models, you must install PyTorch with CUDA support. Follow `PyTorch - Get Started <https://pytorch.org/get-started/locally/>`_ for installation steps. We recommend installing the `PEFT library <https://pypi.org/project/peft/>`_, especially if you plan to perform parameter-efficient fine-tuning or run inference with models fine-tuned using PEFT

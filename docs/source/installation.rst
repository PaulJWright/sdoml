.. _installation:

============
Installation
============


To use ``sdoml``, first install it using pip:

.. code-block:: console

   git clone https://https://github.com/PaulJWright/sdoml.git
   cd sdoml
   pip install -e .

If you would like to access and use the data stored on the Google Cloud Platform, you may need to install the Google Cloud Command Line Interface (`gcloud CLI <https://cloud.google.com/sdk/docs/install>`_). After install, you may need to run the following commands:

.. code-block:: console

   gcloud init
   gcloud auth application-default login

Requirements
============

``sdoml`` requires Python 3.7, or higher.

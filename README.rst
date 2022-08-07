SDOML dataset
---------------------

SDOML is an open-source package for working with the SDOML Dataset (`sdoml.org <https://sdoml.org>`_).

Installation
------------

If you'd like to help develop the SDOML package, or just want to try out the package, you will need to install it from GitHub. The best way to do this is to create a new python virtual environment (either with pipenv or conda). Once you have that virtual environment:

.. code:: bash

  $ git clone https://https://github.com/PaulJWright/sdoml.git
  $ cd sdoml
  $ pip install -e .


To install the optional extras required for testing, this can be performed with pip as below (for ``bash``, and ``zsh``)

.. code:: bash

  $ pip install -e .[test]

.. code:: zsh

  ~ pip install -e '.[test]'

If you would like to access and use the data stored on the Google Cloud Platform, you may need to install the Google Cloud Command Line Interface (`gcloud CLI <https://cloud.google.com/sdk/docs/install>_`).
After install, you may need to run the following commands:

.. code:: bash

  gcloud init
  gcloud auth application-default login

License
-------

This project is Copyright (c) Paul J. Wright and licensed under
the terms of the Apache Software License 2.0 license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.


Contributing
------------

We love contributions! sdoml is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
sdoml based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.

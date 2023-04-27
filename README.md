# my-cokkiecutter

A simple repo template with stuff I use im most repositories.

Currently, it contains a Sphinx documentation template, a `pyproject.toml` template file,
a `.pre-commit-config.yaml` with common settings and a package structure with a Makefile
to quickly create a Python virtual environment.

The virtual environment can be set up quickly by

```
make venv
```

and then activated by

```
source .venv/bin/activate
```

At the moment, it is necessary to manually rewrite all occurences of `my-cookiecutter`
in the repo structure and in `pyproject.toml` to the actual project name.

The documentation can be built (into html) by going to the `docs/source` directory
and calling

```
make html
```

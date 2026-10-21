from setuptools import setup, find_packages

setup(
    name="subshowers",
    version="0.1.0",
    packages=find_packages(include=["subshowers", "subshowers.*"]),
    install_requires=[
        "pandas>3",
        "tables",
        "pyarrow",
        "plotly",
        "scikit-image",
    ],
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydatasci",
    version="0.0.56",
    author="Layne Sadler",
    author_email="layne@pydatasci.org",
    description="End-to-end machine learning on your desktop or server.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pydatasci-repo",
    packages=setuptools.find_packages(),
    install_requires=[
        'appdirs',
        'keras',
        'numpy',
        'pandas',
        'peewee',
        'plotly',
        'pyarrow',
        'scikit-learn',
        'tensorflow',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)

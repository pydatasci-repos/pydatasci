import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydatasci",
    version="0.0.18",
    author="Layne Sadler",
    author_email="layne@pydatasci.org",
    description="End-to-end machine learning on your desktop.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pydatasci-repo",
    packages=setuptools.find_packages(),
    install_requires=[
        'appdirs',
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

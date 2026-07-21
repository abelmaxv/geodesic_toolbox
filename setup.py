from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geodesic_toolbox",
    version="0.0.1",
    author="Théau Blanchard",
    author_email="theau.blanchard@gehealthcare.com",
    description="Differentiable geodesic trajectories, distances and sampling on manifold in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="",
    # project_urls={"Bug Tracker": ""},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.11.9",
        # "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11.9",
    # Core library dependencies (what `import geodesic_toolbox` needs). Loose
    # lower bounds only; exact versions for reproducibility live in
    # requirements.txt / the conda env.
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "tqdm>=4.60",
        "einops>=0.6",
        "networkx>=3.0",
        "torchdiffeq>=0.2",
        "kmedoids>=0.5",
    ],
    extras_require={
        # Running the benchmarks (benchmarks/): `pip install -e ".[benchmarks]"`
        "benchmarks": ["pandas>=2.0", "hamiltorch>=0.4", "matplotlib>=3.6"],
        # Notebooks / animations rendering: `pip install -e ".[viz]"`
        "viz": ["matplotlib>=3.6", "ipykernel>=6.0"],
    },
)

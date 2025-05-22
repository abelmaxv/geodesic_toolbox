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
)

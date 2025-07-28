from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Hybrid-Anime-recommender",
    version="0.1",
    author="Sandeep Hidellarachchi",
    packages=find_packages(),
    install_requires = requirements,
)
from setuptools import setup, find_packages

setup(
    name="xarm_env",
    version="0.1.0",
    install_requires=["gymnasium", "numpy"],
    packages=find_packages(),
)
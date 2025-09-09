from setuptools import find_packages, setup

setup(
    name="mlops_project",
    version="0.0.1",
    author="Your Name",
    author_email="youremail@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
        "boto3"
    ],
)

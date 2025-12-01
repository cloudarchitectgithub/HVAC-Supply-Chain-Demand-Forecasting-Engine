"""
Setup script to install the package in development mode
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="supply-chain-forecast",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered demand forecasting for HVAC supply chain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hvac-demand-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "pandas==2.1.4",
        "numpy==1.26.2",  # FIXED: Match requirements.txt
        "sqlalchemy==2.0.23",
        "prophet==1.1.5",
        "scikit-learn==1.3.2",
        "fastapi==0.109.0",  # FIXED: Match requirements.txt
        "uvicorn==0.27.0",  # FIXED: Match requirements.txt
        "pydantic==2.5.3",  # FIXED: Match requirements.txt
        "requests==2.31.0",
        "streamlit==1.29.0",
        "plotly==5.18.0",
        "python-dateutil==2.8.2",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "forecast-pipeline=run_pipeline:main",
        ],
    },
)
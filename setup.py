from setuptools import setup, find_packages

setup(
    name="nimo",
    version="1.0.5",
    author="NIMO developers",
    license="MIT",
    description='NIMO package',
    packages=["nimo", "nimo.ai_tools", "nimo.input_tools", "nimo.output_tools", "nimo.visualization"],
    install_requires=[
        "Cython",
        "matplotlib",
        "numpy",
        "physbo>=2.0.0",
        "scikit-learn",
        "scipy"
    ]
)

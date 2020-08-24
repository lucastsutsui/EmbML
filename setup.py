from setuptools import setup,find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="embml",
    version="0.0.4",
    author="Lucas Tsutsui da Silva",
    author_email="lucastsui@hotmail.com",
    description="A tool to support using classification models in low-power microcontroller-based hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucastsutsui/embml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'javaobj'
    ],
    license="GPL3",
    keywords=[
        'weka',
        'scikit-learn',
        'microcontroller',
        'classifier',
        'embedded'
    ],
    python_requires='>=2.7',
    include_package_data=True,
)

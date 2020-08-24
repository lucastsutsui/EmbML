import setuptools

setuptools.setup(
    name="embml",
    version="0.0.2",
    author="Lucas Tsutsui da Silva",
    author_email="lucastsui@hotmail.com",
    description="A tool to support using classification models in low-power microcontroller-based hardware",
    url="https://github.com/lucastsutsui/embml",
    packages=setuptools.find_packages(),
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
)

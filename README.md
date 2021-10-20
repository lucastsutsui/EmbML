
# EmbML

EmbML is a tool written in Python to automatically convert off-board-trained models into C++ (default option) or C source code files that can be compiled and executed in low-power microcontrollers. The main goal of EmbML is to produce classifier source codes that will run specifically in resource-constrained hardware systems, using bare metal programming.

This tool takes as input a classification model that was trained in a desktop or server computer using WEKA or scikit-learn libraries. EmbML is responsible for converting the input model into a carefully crafted code in C or C++ with support for embedded hardware, such as the avoidance of unnecessary use of SRAM memory and implementation of fixed-point operations for non-integer numbers. 

## Input Models

EmbML accepts a trained model through the file that contains its serialized object. For instance, a classification model, built with WEKA, shall be serialized into a file using the _ObjectOutputStream_ and _FileOutputStream_ classes (available in Java). [Example of saving a WEKA model using its GUI.](https://machinelearningmastery.com/save-machine-learning-model-make-predictions-weka/).

As for the scikit-learn models, they shall be serialized using the _dump_ function, from _pickle_ module. An example is provided in <https://scikit-learn.org/stable/modules/model_persistence.html>.

## Supported Classification Models

`embml` supports off-board-trained classifiers from the following classes:

* From WEKA:
	* _MultilayerPerceptron_ for MLP classifiers;
	* _Logistic_ for logistic regression classifiers;
	* _SMO_ for SVM classifiers -- with linear, polynomial, and RBF kernels;
	* _J48_ for decision tree classifier.
* From scikit-learn:
	* _MLPClassifier_ for MLP classifiers;
	* _LogisticRegression_ for logistic regression classifiers;
	* _LinearSVC_ for SVM classifiers with linear kernel;
	* _SVC_ for SVM classifiers -- with polynomial and RBF kernels;
	* _DecisionTreeClassifier_ for decision tree models.

## Installation

You can install `embml` from [PyPi](https://pypi.org/project/embml/):

```python
pip install embml
```

This tool is supported on Python 2.7 and Python 3.7 versions, and depends on the `javaobj` library (<https://pypi.org/project/javaobj-py3/>).

## How To Use

```python
import embml

# For scikit-learn models
embml.sklearnModel(inputModel, outputFile, opts)

# For WEKA models
embml.wekaModel(inputModel, outputFile, opts)
		
# opts can include:
#	-rules: to generate a decision tree classifier code using a representation with if-then-else statements.
#	-fxp <n> <m>: to generate a classifier code that uses fixed-point format to perform real number operations. In this case, <n> is the number of integer bits and <m> is the number of fractional bits in the Qn.m format. Note that n + m + 1 must be equal to 32, 16, or 8, since that one bit is used to represent signed numbers.
#	-approx: to generate an MLP classifier code that employs an approximation to substitute the sigmoid as an activation function in the neurons.
#	-pwl <x>: to generate an MLP classifier code that employs a piecewise approximation to substitute the sigmoid as an activation function in the neurons. In this case, <x> must be equal to 2 (to use an 2-point PWL approximation) or 4 (to use an 4-point PWL approximation).

# Examples of generating decision tree classifier codes using if-then-else format.
embml.wekaModel(inputDecisionTreeModel, outputFile, opts='-rules')
embml.sklearnModel(inputDecisionTreeModel, outputFile, opts='-rules')

# Examples of generating classifier codes in C programming language.
embml.wekaModel(inputModel, outputFile, opts='-c')
embml.sklearnModel(inputModel, outputFile, opts='-c')

# Examples of generating classifier codes using fixed-point formats.
embml.wekaModel(inputModel, outputFile, opts='-fxp 21 10') # Q21.10
embml.sklearnModel(inputModel, outputFile, opts='-fxp 21 10') # Q21.10
embml.wekaModel(inputModel, outputFile, opts='-fxp 11 4') # Q11.4
embml.sklearnModel(inputModel, outputFile, opts='-fxp 11 4') # Q11.4
embml.wekaModel(inputModel, outputFile, opts='-fxp 5 2') # Q5.2
embml.sklearnModel(inputModel, outputFile, opts='-fxp 5 2') # Q5.2

# Examples of generating MLP classifier codes using an approximation function.
embml.wekaModel(inputMlpModel, outputFile, opts='-approx')
embml.sklearnModel(inputMlpModel, outputFile, opts='-approx')

# Examples of generating MLP classifier codes using PWL approximations.
embml.wekaModel(inputMlpModel, outputFile, opts='-pwl 2')
embml.sklearnModel(inputMlpModel, outputFile, opts='-pwl 2')
embml.wekaModel(inputMlpModel, outputFile, opts='-pwl 4')
embml.sklearnModel(inputMlpModel, outputFile, opts='-pwl 4')

# It is also possible to combine some options:	
embml.wekaModel(inputMlpModel, outputFile, opts='-fxp 21 10 -pwl 2')
embml.sklearnModel(inputMlpModel, outputFile, opts='-fxp 21 10 -pwl 2')
embml.wekaModel(inputDecisionTreeModel, outputFile, opts='-fxp 21 10 -rules')
embml.sklearnModel(inputDecisionTreeModel, outputFile, opts='-fxp 21 10 -rules')
```

## Fixed-point library

If you decide to generate a classifier code using a fixed-point format, you need to include the `FixedNum.h` library available at [https://github.com/lucastsutsui/EmbML](https://github.com/lucastsutsui/EmbML).

## Citation

If you use this tool on a scientific work, we kindly ask you to use the following reference:

```tex
@inproceedings{da2019embml,
  title={EmbML Tool: supporting the use of supervised learning algorithms in low-cost embedded systems},
  author={da Silva, Lucas Tsutsui and Souza, Vinicius MA and Batista, Gustavo EAPA},
  booktitle={2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)},
  pages={1633--1637},
  year={2019},
  organization={IEEE}
}
```


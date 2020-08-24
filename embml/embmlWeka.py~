
from __future__ import print_function
from .wekaModels import embml_weka_J48
from .wekaModels import embml_weka_Logistic
from .wekaModels import embml_weka_MultilayerPerceptron
from .wekaModels import embml_weka_SMO_LinearKernel
from .wekaModels import embml_weka_SMO_PolyKernel
from .wekaModels import embml_weka_SMO_RBFKernel


def recoverWeka(model, opts):
    if 'weka.classifiers.trees.J48' in str(model):
        return embml_weka_J48.recover(model, opts)
    elif 'weka.classifiers.functions.Logistic' in str(model):
        return embml_weka_Logistic.recover(model, opts)
    elif 'weka.classifiers.functions.MultilayerPerceptron' in str(model):
        return embml_weka_MultilayerPerceptron.recover(model, opts)
    elif 'weka.classifiers.functions.SMO' in str(model):
        if 'weka.classifiers.functions.supportVector.RBFKernel' in str(vars(model)['m_kernel']):
            return embml_weka_SMO_RBFKernel.recover(model, opts)
        elif 'weka.classifiers.functions.supportVector.PolyKernel' in str(vars(model)['m_kernel']):
            if vars(model)['m_KernelIsLinear']:
                return embml_weka_SMO_LinearKernel.recover(model, opts)
            else:
                return embml_weka_SMO_PolyKernel.recover(model, opts)
        else:
            print ("Error: SVM kernel not supported")
            exit(1)
    else:
        print ("Error: classification model " + str(model) + " not supported")
        exit(1)
    

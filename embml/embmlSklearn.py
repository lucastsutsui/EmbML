
from __future__ import print_function
from .sklearnModels import embml_sklearn_DecisionTreeClassifier
from .sklearnModels import embml_sklearn_LogisticRegression
from .sklearnModels import embml_sklearn_LinearSVC
from .sklearnModels import embml_sklearn_MLPClassifier
from .sklearnModels import embml_sklearn_SVC_Kernel

def recoverSklearn(model, opts):
    if 'DecisionTreeClassifier' in str(model):
        return embml_sklearn_DecisionTreeClassifier.recover(model, opts)
    elif 'LogisticRegression' in str(model):
        return embml_sklearn_LogisticRegression.recover(model, opts)
    elif 'MLPClassifier' in str(model):
        return embml_sklearn_MLPClassifier.recover(model, opts)
    elif 'LinearSVC' in str(model):
        return embml_sklearn_LinearSVC.recover(model, opts)
    elif 'SVC' in str(model):
        return embml_sklearn_SVC_Kernel.recover(model, opts)
    else:
        print ("Error: classification model not supported:\n\n" + str(model))
        exit(1)
    



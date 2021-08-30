
from ..utils import utils
import numpy as np

class sklearn_DecisionTreeClassifier:

    def process(self, model, opts):
        TREE_LEAF = -1
        
        self.children_left = model.tree_.children_left.tolist()
        self.children_right = model.tree_.children_right.tolist()
        value = model.tree_.value
        self.threshold = model.tree_.threshold.tolist()
        self.feature = model.tree_.feature.tolist()
        
        self.inputSize = model.tree_.n_features
        self.treeSize = len(model.tree_.threshold)

        for i in range(len(self.feature)):
            if self.children_left[i] == TREE_LEAF and\
               self.children_right[i] == TREE_LEAF:
                self.feature[i] = np.argmax(value[i][0])

        self.classes = list(map(int, model.classes_.tolist()))

        if opts['useFxp']:
            self.threshold = [utils.toFxp(_i, opts) \
                                    for _i in self.threshold]
        

def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")
    
    # Classify function
    funcs += utils.write_func_init("int", "classify") + \
    utils.write_dec("int", "i", initValue="ROOT", tabs=1) + \
    utils.write_while("i != -1 && children_left[i] != -1 && children_right[i] != -1", \
                      tabs=1) + \
    utils.write_if("instance[feature[i]] <= threshold[i]", tabs=2) + \
    utils.write_attribution("i", "children_left[i]", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_else(tabs=2) + \
    utils.write_attribution("i", "children_right[i]", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("classes[feature[i]]", tabs=1) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(classifier.feature), \
                    "feature[LEN_TREE]", \
                    initValue=utils.toStr(classifier.feature)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "threshold[LEN_TREE]", \
                    initValue=utils.toStr(classifier.threshold)) + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(classifier.children_left), \
                    "children_left[LEN_TREE]", \
                    initValue=utils.toStr(classifier.children_left)) + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(classifier.children_right), \
                    "children_right[LEN_TREE]", \
                    initValue=utils.toStr(classifier.children_right)) + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(classifier.classes), \
                    "classes[NUM_CLASSES]", \
                    initValue=utils.toStr(classifier.classes)) + '\n'

    # Definition of constant values
    defs += utils.write_define("ROOT", "0") + \
    utils.write_define("NUM_CLASSES", str(len(classifier.classes))) + \
    utils.write_define("LEN_TREE", str(classifier.treeSize)) + \
    utils.write_define("INPUT_SIZE", str(classifier.inputSize))
        
    # Include of libraries
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs)

def generate_rules(node, classifier, tabs, opts):
    if node == -1 or \
       classifier.children_left[node] == -1 or \
       classifier.children_right[node] == -1:
        return utils.write_ret(str(classifier.classes[classifier.feature[node]]), \
                               tabs=tabs)
    return utils.write_if("instance[" + \
                          str(classifier.feature[node]) + \
                          "] <= " + \
                          (classifier.threshold[node] if opts['useFxp'] else ("%.10f" % classifier.threshold[node])), \
                          tabs=tabs) + \
        generate_rules(classifier.children_left[node], classifier, tabs+1, opts) + \
        utils.write_end(tabs=tabs) + \
        utils.write_else(tabs=tabs) + \
        generate_rules(classifier.children_right[node], classifier, tabs+1, opts) + \
        utils.write_end(tabs=tabs)

def write_output_rules(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")
    
    # Classify function
    funcs += utils.write_func_init("int", "classify") + \
    generate_rules(0, classifier, 1, opts) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n'

    # Definition of constant values
    defs += utils.write_define("NUM_CLASSES", str(len(classifier.classes))) + \
    utils.write_define("LEN_TREE", str(classifier.treeSize)) + \
    utils.write_define("INPUT_SIZE", str(classifier.inputSize))
        
    # Include of libraries
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs)

def recover(model, opts):
    classifier = sklearn_DecisionTreeClassifier()
    classifier.process(model, opts)

    if opts['rules']:
        return write_output_rules(classifier, opts)
    return write_output(classifier, opts)

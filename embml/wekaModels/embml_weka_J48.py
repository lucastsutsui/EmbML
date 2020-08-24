
from __future__ import print_function
from ..utils import utils

class weka_J48:

    def dfsLengthTree(self, node):
        # Recovers the length of the tree
        if vars(node)['m_isLeaf']:
            return 1
        return sum([self.dfsLengthTree(i) for i in vars(node)['m_sons']]) + 1

    def dfs(self, node, index):
        # Recovers the structure of the tree
        i = index

        if vars(node)['m_isLeaf']:
            self.m_isLeaf[i] = True

        m_localModel = vars(node)['m_localModel']
        m_distribution = vars(m_localModel)['m_distribution']
        self.m_perClass[i] = vars(m_distribution)['m_perClass']
        self.m_perBag[i] = vars(m_distribution)['m_perBag']
        self.m_perClassPerBag[i] = vars(m_distribution)['m_perClassPerBag']
        self.m_isEmpty[i] = vars(node)['m_isEmpty']
        self.m_totaL[i] = vars(m_distribution)['totaL']
    
        if 'm_attIndex' in vars(m_localModel):
            self.m_attIndex[i] = vars(m_localModel)['m_attIndex']
        if 'm_splitPoint' in vars(m_localModel):
            self.m_splitPoint[i] = vars(m_localModel)['m_splitPoint']
            
        if type(vars(node)['m_sons']) != type(None):
            for son in vars(node)['m_sons']:
                i, childIndex = self.dfs(son, i + 1)
                self.tree[index].append(childIndex)
        
        return (i, index)

    def getProb1(self, i):
        probPerClass = []

        if self.m_useLaplace:
            probPerClass = [((self.m_perClass[i][j] + 1.0) / float(self.m_totaL[i] + self.m_numClasses))\
                        for j in range(len(self.m_perClass[i]))]
        else:
            if self.m_totaL[i] != 0:
                probPerClass = [(self.m_perClass[i][j] / float(self.m_totaL[i]))\
                                for j in range(len(self.m_perClass[i]))]
            else:
                probPerClass = [0.0 for j in range(len(self.m_perClass[i]))]
                
        return probPerClass.index(max(probPerClass)) + self.attOffset

    def getProb2(self, i, j):
        probPerClass = []
        
        if self.m_perBag[i][j] > 0:
            if self.m_useLaplace:
                probPerClass = [((self.m_perClassPerBag[i][j][k] + 1.0) /\
                                 float(self.m_perBag[i][j] + self.m_numClasses))\
                                for k in range(len(self.m_perClassPerBag[i][j]))]
            else:
                probPerClass = [(self.m_perClassPerBag[i][j][k] /\
                                 float(self.m_perBag[i][j]))\
                                for k in range(len(self.m_perClassPerBag[i][j]))]
        else:
            return self.getProb1(i)
        
        return probPerClass.index(max(probPerClass)) + self.attOffset
        

    def setPrediction(self, i):
        # Pre-calculate predicted classes for each leaf
        if self.m_isLeaf[i]:
            self.m_attIndex[i] = self.getProb1(i)
        else:
            for j in range(len(self.tree[i])):
                if self.m_isEmpty[i]:
                    self.m_attIndex[i] = getProb2(i, j)
                else:
                    self.setPrediction(self.tree[i][j])

    def process(self, model):
        self.m_useLaplace = vars(model)['m_useLaplace']

        # Recover index for class attribute
        try:
            self.m_classIndex = vars(vars(vars(model)['m_root'])['m_train'])['m_ClassIndex']
        except:
            print ("Error: can't recover class index")
            exit(1)

        # Recover attribute names
        try:
            inputs = vars(vars(vars(vars(model)['m_root'])['m_train'])\
                          ['m_NamesToAttributeIndices'])['annotations'][1:]
            self.attributes = ['' for _ in range((len(inputs) // 2) - 1)]
            for i in range(len(inputs)):
                if i % 2 == 0:
                    continue
                index = vars(inputs[i])['value']
                if index != self.m_classIndex:
                    self.attributes[index - (1 if index > self.m_classIndex else 0)] = inputs[i - 1]
        except:
            print ("Error: can't recover attribute names")
            exit(1)

        # Recover class names
        try:
            outputs = vars(vars(vars(vars(vars(vars(vars(model)['m_root'])['m_train'])\
                                ['m_Attributes'])['annotations'][1:][self.m_classIndex])\
                                ['m_AttributeInfo'])['m_Hashtable'])['annotations'][1:]
            self.classes = ['' for _ in range(len(outputs) // 2)]
            for i in range(len(outputs)):
                if i % 2 == 0:
                    continue
                self.classes[vars(outputs[i])['value']] = outputs[i - 1]
        except:
            print ("Error: can't recover class names")
            exit(1)

        self.m_numClasses = len(self.classes)

        self.lenTree = self.dfsLengthTree(vars(model)['m_root'])

        self.tree = [[] for _ in range(self.lenTree)]
        self.m_isLeaf = [False for _ in range(self.lenTree)]
        self.m_attIndex = [-1 for _ in range(self.lenTree)]
        self.m_splitPoint = [-1 for _ in range(self.lenTree)]
        self.m_isEmpty = [False for _ in range(self.lenTree)]
        self.m_totaL = [0 for _ in range(self.lenTree)]
        self.m_perClass = [[] for _ in range(self.lenTree)]
        self.m_perBag = [[] for _ in range(self.lenTree)]
        self.m_perClassPerBag = [[] for _ in range(self.lenTree)]
    
        self.dfs(vars(model)['m_root'], 0)
    
        # Save the indexes for empty nodes
        self.emptyIndex = [i for i in range(len(self.m_isEmpty)) if self.m_isEmpty[i]]

        for i in range(len(self.m_attIndex)):
            if self.m_isEmpty[i]:
                self.m_attIndex[i] = -2

        self.attOffset = len(self.attributes) + 2
        
        self.setPrediction(0)
        
        for i in range(len(self.tree)):
            if len(self.tree[i]) == 0:
                self.tree[i] = [-1, -1]

def write_output(weka_j48, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")
    
    # Classify function
    funcs += utils.write_output_classes(weka_j48.classes) + '\n' + \
    utils.write_func_init("int", "classify") + \
    utils.write_dec("int", "i", initValue="M_ROOT", tabs=1) + \
    utils.write_while("i != -1 && m_attIndex[i] < ATT_OFFSET", tabs=1) + \
    utils.write_attribution("i", \
                            "tree[i][(instance[m_attIndex[i]] <= m_splitPoint[i] ? 0 : 1)]", \
                            tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("(m_attIndex[i] - ATT_OFFSET)", tabs=1) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    if opts['useFxp']:
        weka_j48.m_splitPoint = [utils.toFxp(_i, opts) for _i in weka_j48.m_splitPoint]
        
    decls += utils.write_attributes(weka_j48.attributes) + '\n' + \
    utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(weka_j48.m_attIndex), \
                    "m_attIndex[LEN_TREE]", \
                    initValue=utils.toStr(weka_j48.m_attIndex)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "m_splitPoint[LEN_TREE]", \
                    initValue=utils.toStr(weka_j48.m_splitPoint)) + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(weka_j48.tree), \
                    "tree[LEN_TREE][2]", \
                    initValue=utils.toStr(weka_j48.tree))

    # Definition of constant values
    defs += utils.write_define("M_ROOT", "0") + \
    utils.write_define("NUM_CLASSES", str(weka_j48.m_numClasses)) + \
    utils.write_define("CLASS_INDEX", str(weka_j48.m_classIndex)) + \
    utils.write_define("LEN_TREE", str(weka_j48.lenTree)) + \
    utils.write_define("INPUT_SIZE", str(len(weka_j48.attributes))) + \
    utils.write_define("ATT_OFFSET", str(weka_j48.attOffset))
        
    # Include of libraries
    incls += utils.write_include("<Arduino.h>")
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs)

def generate_rules(node, weka_j48, tabs, opts):
    if node == -1 or weka_j48.m_attIndex[node] >= weka_j48.attOffset:
        return utils.write_ret(str(weka_j48.m_attIndex[node] - weka_j48.attOffset), tabs=tabs)
    return utils.write_if("instance[" + \
                          str(weka_j48.m_attIndex[node]) + \
                          "] <= " + \
                          (utils.toFxp(weka_j48.m_splitPoint[node], opts) \
                           if opts['useFxp'] \
                           else ("%.10f" % weka_j48.m_splitPoint[node])), \
                          tabs=tabs) + \
        generate_rules(weka_j48.tree[node][0], weka_j48, tabs+1, opts) + \
        utils.write_end(tabs=tabs) + \
        utils.write_else(tabs=tabs) + \
        generate_rules(weka_j48.tree[node][1], weka_j48, tabs+1, opts) + \
        utils.write_end(tabs=tabs)

def write_output_rules(weka_j48, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")
    
    # Classify function
    funcs += utils.write_output_classes(weka_j48.classes) + '\n' + \
    utils.write_func_init("int", "classify") + \
    generate_rules(0, weka_j48, 1, opts) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_attributes(weka_j48.attributes) + '\n' + \
    utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n'

    # Definition of constant values
    defs += utils.write_define("NUM_CLASSES", str(weka_j48.m_numClasses)) + \
    utils.write_define("CLASS_INDEX", str(weka_j48.m_classIndex)) + \
    utils.write_define("INPUT_SIZE", str(len(weka_j48.attributes)))
        
    # Include of libraries
    incls += utils.write_include("<Arduino.h>")
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs)

def recover(model, opts):
    classifier = weka_J48()
    classifier.process(model)

    if opts['rules']:
        return write_output_rules(classifier, opts)
    return write_output(classifier, opts)

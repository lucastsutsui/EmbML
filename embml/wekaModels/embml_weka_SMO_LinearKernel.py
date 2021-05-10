
from __future__ import print_function
from ..utils import utils
#import javaobj

class weka_SMO_LinearKernel:

    def process(self, model, opts):

        self.m_numClasses = vars(vars(vars(vars(model)\
                        ['m_classAttribute'])['m_AttributeInfo'])\
                        ['m_Values'])['size']
        self.m_classIndex = vars(model)['m_classIndex']

        self.minArray = vars(vars(model)['m_Filter'])['m_MinArray']
        self.maxArray = vars(vars(model)['m_Filter'])['m_MaxArray']

        # Recover attribute names
        try:
            inputs = vars(vars(vars(vars(vars(model)\
                    ['m_Filter'])['m_InputRelAtts'])\
                    ['m_Data'])['m_NamesToAttributeIndices'])\
                    ['annotations'][1:]
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
            outputs = vars(vars(vars(vars(model)\
                        ['m_classAttribute'])['m_AttributeInfo'])\
                        ['m_Hashtable'])['annotations'][1:]
        except:
            print ("Error: can't recover class names")
            exit(1)
        self.classes = ['' for _ in range(len(outputs) // 2)]
        for i in range(len(outputs)):
            if i % 2 == 0:
                continue
            self.classes[vars(outputs[i])['value']] = outputs[i - 1]

        m_classifiers = vars(model)['m_classifiers']

        # Recover SVM binary classifiers
        self.new_classifiers = [[] for _ in m_classifiers]
        for i in range(len(m_classifiers)):
            for j in range(len(m_classifiers[i])):
                if "None" in str(type(m_classifiers[i][j])):
                    continue
                self.new_classifiers[j].append(m_classifiers[i][j])

        # Recover m_sparseWeights and m_b arrays
        # Convert to fixed-point, if necessary
        self.m_sparseWeights = []
        self.m_b = []
        for i in range(len(self.new_classifiers)):
            tmp_sparseWeights = []
            tmp_m_b = []

            for j in range(len(self.new_classifiers[i])):
                tmp_sparseWeights += vars(self.new_classifiers[i][j])['m_sparseWeights']
                tmp_m_b.append(vars(self.new_classifiers[i][j])['m_b'])
                
            if opts['useFxp']:
                tmp_sparseWeights = [utils.toFxp(tmp_sparseWeights[_i], opts) \
                                     for _i in range(len(tmp_sparseWeights))]
                tmp_m_b = [utils.toFxp(tmp_m_b[_i], opts) \
                                     for _i in range(len(tmp_m_b))]
                
            self.m_b.append(tmp_m_b)
            self.m_sparseWeights.append(tmp_sparseWeights)

        # Convert minArray and maxArray to fixed-point, if necessary
        if opts['useFxp']:
            self.minArray = [utils.toFxp(self.minArray[_i], opts) \
                                     for _i in range(len(self.minArray))]
            self.maxArray = [utils.toFxp(self.maxArray[_i], opts) \
                                     for _i in range(len(self.maxArray))]

def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'
    inits = '\nvoid initConnections(){\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")
    sizeMinArray = len(classifier.minArray)
    sizeMaxArray = len(classifier.maxArray)

    # SVMOutput function
    funcs += utils.write_func_init(decType, "SVMOutput", args="int i, int j") + \
    utils.write_dec(decType, "result", \
                    initValue=(utils.toFxp(0.0, opts)\
                               if opts['useFxp'] else \
                               "0.0"), tabs=1) + \
    utils.write_for("p1 = 0", "p1 < INPUT_SIZE", "p1++", tabs=1) + \
    utils.write_if("p1 != CLASS_INDEX", tabs=2) + \
    utils.write_attribution("result", \
                            ("fxp_sum(result, fxp_mul(instance[p1], m_sparseWeights[i][(j * INPUT_SIZE) + p1]))"\
                            if opts['useFxp'] else \
                            "instance[p1] * m_sparseWeights[i][(j * INPUT_SIZE) + p1]"), \
                            op=('' if opts['useFxp'] else '+'), \
                            tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_attribution("result", \
                            ("fxp_diff(result, m_b[i][j])" \
                             if opts['useFxp'] else \
                             "m_b[i][j]"), \
                            op=('' if opts['useFxp'] else '-'), \
                            tabs=1) + \
    utils.write_ret("result", tabs=1) + \
    utils.write_end(tabs=0)
    
    # Classify function
    funcs += utils.write_output_classes(classifier.classes) + '\n' + \
    utils.write_func_init("int", "classify") + \
    utils.write_for("i = 0", "i <= INPUT_SIZE", "i++", tabs=1) + \
    utils.write_if("maxArray[i] == minArray[i] || minArray[i] == " + \
                   ("INF_POS" \
                    if opts['useFxp'] else \
                    "NAN"), tabs=2) + \
    utils.write_attribution("instance[i]", \
                            (utils.toFxp(0.0, opts) \
                             if opts['useFxp'] else \
                             "0"), \
                            tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_else(tabs=2) + \
    utils.write_attribution("instance[i]", \
                ("fxp_div(fxp_diff(instance[i], minArray[i]), fxp_diff(maxArray[i], minArray[i]))" \
                 if opts['useFxp'] else \
                 "(instance[i] - minArray[i]) / (maxArray[i] - minArray[i])"), \
                tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec("int", "result[NUM_CLASSES]", \
                    initValue="{0}", tabs=1) + \
    utils.write_for("i = 1", "i < NUM_CLASSES", "i++", tabs=1) + \
    utils.write_for("j = 0", "j < i", "j++", tabs=2) + \
    utils.write_dec(decType, "output", \
                    initValue="SVMOutput(i, j)", tabs=3) + \
    utils.write_if("output > 0", tabs=3) + \
    utils.write_inc_dec("result[i]++", tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_else(tabs=3) + \
    utils.write_inc_dec("result[j]++", tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec("int", "indMax", \
                    initValue="0", tabs=1) + \
    utils.write_for("i = 1", "i < NUM_CLASSES", "i++", tabs=1) + \
    utils.write_if("result[i] > result[indMax]", tabs=2) + \
    utils.write_attribution("indMax", "i", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("indMax", tabs=1) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_attributes(classifier.attributes) + '\n' + \
    utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec("const " + decType, \
                    "minArray[" + str(sizeMinArray) + "]", \
                    initValue=utils.toStr(classifier.minArray)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "maxArray[" + str(sizeMaxArray) + "]", \
                    initValue=utils.toStr(classifier.maxArray)) + '\n'

    ### Initialize m_sparseWeights array
    decls += utils.write_dec("const " + decType, \
            "*m_sparseWeights[" + str(len(classifier.m_sparseWeights)) + "]") + '\n'
    
    for i in range(len(classifier.m_sparseWeights)):
        if len(classifier.m_sparseWeights[i]) == 0:
            inits += utils.write_attribution("m_sparseWeights[" + str(i) + "]", \
                                            "NULL", tabs=1)
            continue
        
        decls += utils.write_dec("const " + decType, \
                "tmp_sparseWeights" + str(i) + \
                "[" + str(len(classifier.m_sparseWeights[i])) + "]", \
                initValue=utils.toStr(classifier.m_sparseWeights[i]))
    
        inits += utils.write_attribution("m_sparseWeights[" + str(i) + "]", \
                                         "tmp_sparseWeights" + str(i), tabs=1)
        
    ### initialize m_b array
    decls += utils.write_dec("const " + decType, \
                    "*m_b[" + str(len(classifier.m_b)) + ']') + '\n'
    
    for i in range(len(classifier.m_b)):
        if len(classifier.m_b[i]) == 0:
            inits += utils.write_attribution("m_b[" + str(i) + "]", \
                                             "NULL", tabs=1)
            continue
        
        decls += utils.write_dec("const " + decType, \
                    "tmp_m_b" + str(i) + \
                    "[" + str(len(classifier.m_b[i])) + "]", \
                    initValue=utils.toStr(classifier.m_b[i]))
    
        inits += utils.write_attribution("m_b[" + str(i) + "]", \
                                            "tmp_m_b" + str(i), tabs=1)
        
    inits += "}\n"

    # Definition of constant values
    defs += utils.write_define("INPUT_SIZE", str(len(classifier.attributes))) + \
    utils.write_define("NUM_CLASSES", str(len(classifier.classes))) + \
    utils.write_define("CLASS_INDEX", str(classifier.m_classIndex))
        
    # Include of libraries
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs + inits)

def recover(model, opts):
    classifier = weka_SMO_LinearKernel()
    classifier.process(model, opts)

    return write_output(classifier, opts)


from __future__ import print_function
from ..utils import utils
#import javaobj

class weka_SMO_PolyKernel:

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
        except:
            print ("Error: can't recover attribute names")
            exit(1)
        self.attributes = ['' for _ in range((len(inputs) // 2) - 1)]
        for i in range(len(inputs)):
            if i % 2 == 0:
                continue
            index = vars(inputs[i])['value']
            if index != self.m_classIndex:
                self.attributes[index - (1 if index > self.m_classIndex else 0)] = inputs[i - 1]

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
        self.new_classifiers = [[] for _i in m_classifiers]
        for i in range(len(m_classifiers)):
            for j in range(len(m_classifiers[i])):
                if "None" in str(type(m_classifiers[i][j])):
                    continue
                self.new_classifiers[j].append(m_classifiers[i][j])
                
        self.m_exponent = vars(vars(self.new_classifiers[1][0])\
                            ['m_kernel'])['m_exponent']
        self.m_lowerOrder = vars(vars(self.new_classifiers[1][0])\
                            ['m_kernel'])['m_lowerOrder']

        # Recover classifier variables
        self.selectedIndices = []
        self.m_classAlpha = []
        self.m_size = []
        self.m_AttValues = []
        self.m_b = []

        for i in range(len(self.new_classifiers)):
            self.selectedIndices.append([])
            self.m_classAlpha.append([])
            self.m_AttValues.append([])
            self.m_b.append([])
            self.m_size.append([])

            for j in range(len(self.new_classifiers)):
                if j < len(self.new_classifiers[i]):
                    self.m_b[i].append(vars(self.new_classifiers[i][j])['m_b'])
                else:
                    self.m_b[i].append(0)

            if len(self.new_classifiers[i]) == 0:
                continue

            for j in range(len(self.new_classifiers[i])):
                self.selectedIndices[i].append([])
                self.m_AttValues[i].append([])
                
                rawArrayAlpha = vars(self.new_classifiers[i][j])['m_alpha']
                rawArrayClass = vars(self.new_classifiers[i][j])['m_class']
                rawAttValues = vars(vars(vars(vars(self.new_classifiers[i][j])\
                                ['m_kernel'])['m_data'])['m_Instances'])\
                                ['annotations'][1:]
                
                for k in range(len(rawArrayAlpha)):
                    classAlpha = (rawArrayAlpha[k] * rawArrayClass[k])
                    
                    # Ignore indices in which (alpha * class) == 0
                    if classAlpha != 0.0:
                        self.selectedIndices[i][j].append(k)
                        self.m_classAlpha[i].append(classAlpha)
                        self.m_AttValues[i][j] += vars(rawAttValues[k])['m_AttValues']

            self.m_size[i] += [len(_j) for _j in self.selectedIndices[i]]
            for j in range(1, len(self.m_size[i])):
                self.m_size[i][j] += self.m_size[i][j - 1]

        # Convert arrays to fixed-point, if necessary
        if opts['useFxp']:
            self.minArray = [utils.toFxp(self.minArray[_i], opts) \
                             for _i in range(len(self.minArray))]
            self.maxArray = [utils.toFxp(self.maxArray[_i], opts) \
                             for _i in range(len(self.maxArray))]
            self.m_classAlpha = [[utils.toFxp(self.m_classAlpha[_i][_j], opts) \
                                  for _j in range(len(self.m_classAlpha[_i]))]\
                                 for _i in range(len(self.m_classAlpha))]
            self.m_AttValues = [[[utils.toFxp(self.m_AttValues[_i][_j][_k], opts) \
                                  for _k in range(len(self.m_AttValues[_i][_j]))]\
                                 for _j in range(len(self.m_AttValues[_i]))]\
                                for _i in range(len(self.m_AttValues))]
            self.m_b = [[utils.toFxp(self.m_b[_i][_j], opts) \
                         for _j in range(len(self.m_b[_i]))]\
                        for _i in range(len(self.m_b))]

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
                    initValue=(utils.toFxp(0.0, opts) \
                               if opts['useFxp'] else \
                               "0.0"), \
                    tabs=1)
    
    if opts['C']: 
        funcs += utils.write_dec("int", "k", tabs=1) + \
                utils.write_dec("int", "p1", tabs=1)
    
    funcs += utils.write_for("k = 0", \
                    "k < (m_size[i][j] - (j == 0 ? 0 : m_size[i][j - 1]))", \
                    "k++", tabs=1, inC=opts['C']) + \
    utils.write_dec(decType, "resultAux", \
                    initValue=(utils.toFxp(0.0, opts) \
                               if opts['useFxp'] else \
                               "0.0"), \
                    tabs=2) + \
    utils.write_for("p1 = 0", "p1 <= INPUT_SIZE", \
                    "p1++", tabs=2, inC=opts['C']) + \
    utils.write_if("p1 != CLASS_INDEX", tabs=3) + \
    utils.write_attribution("resultAux", \
        ("fxp_sum(resultAux, fxp_mul(instance[p1], m_AttValues[i][j][(k * (INPUT_SIZE + 1)) + p1]))" \
         if opts['useFxp'] else \
         "instance[p1] * m_AttValues[i][j][(k * (INPUT_SIZE + 1)) + p1]"), \
        op=('' if opts['useFxp'] else '+'), \
        tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_end(tabs=2)

    if classifier.m_lowerOrder:
        funcs += utils.write_attribution("resultAux", \
                            (("= fxp_sum(resultAux, " + utils.toFxp(1.0, opts) + ")") \
                             if opts['useFxp'] else \
                             "1.0"), \
                            op=('' if opts['useFxp'] else '+'), tabs=2)
    if classifier.m_exponent != 1.0:
        funcs += utils.write_attribution("resultAux", \
                            ("fxp_pow(resultAux, M_EXPONENT)" \
                             if opts['useFxp'] else \
                             "powf(resultAux, M_EXPONENT)"), tabs=2)
        
    funcs += utils.write_attribution("result", \
        ("fxp_sum(result, fxp_mul(m_class_alpha[i][(j == 0 ? 0 : m_size[i][j - 1]) + k], resultAux))" \
         if opts['useFxp'] else \
         "m_class_alpha[i][(j == 0 ? 0 : m_size[i][j - 1]) + k] * resultAux"), \
        op=('' if opts['useFxp'] else '+'), \
        tabs=2) + \
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
    utils.write_func_init("int", "classify")
    
    if opts['C']: 
        funcs += utils.write_dec("int", "i", tabs=1) + \
                utils.write_dec("int", "j", tabs=1)
    
    funcs += utils.write_for("i = 0", "i <= INPUT_SIZE", "i++", tabs=1, inC=opts['C']) + \
    utils.write_if("maxArray[i] == minArray[i] || minArray[i] == " + \
                   ("INF_POS" if opts['useFxp'] else "NAN"), tabs=2) + \
    utils.write_attribution("instance[i]", \
                            (utils.toFxp(0.0, opts) \
                             if opts['useFxp'] else \
                             "0"), tabs=3) + \
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
    utils.write_for("i = 1", "i < NUM_CLASSES", "i++", tabs=1, inC=opts['C']) + \
    utils.write_for("j = 0", "j < i", "j++", tabs=2, inC=opts['C']) + \
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
    utils.write_for("i = 1", "i < NUM_CLASSES", "i++", tabs=1, inC=opts['C']) + \
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

    ### Initialize m_class_alpha array
    decls += utils.write_dec("const " + decType, \
                    "*m_class_alpha[" + str(len(classifier.m_classAlpha)) + \
                    "]") + '\n'
    
    for i in range(len(classifier.m_classAlpha)):
        if len(classifier.m_classAlpha[i]) == 0:
            inits += utils.write_attribution("m_class_alpha[" + str(i) + "]", \
                                             "NULL", tabs=1) + '\n'
            continue
        decls += utils.write_dec("const " + decType, \
                    "tmp0_class_alpha" + str(i) + "[" + \
                    str(len(classifier.m_classAlpha[i])) + "]", \
                    initValue=utils.toStr(classifier.m_classAlpha[i])) + '\n'
        inits += utils.write_attribution("m_class_alpha[" + str(i) + "]", \
                                         "tmp0_class_alpha" + str(i), \
                                         tabs=1) + '\n'

    ### Initialize m_size array
    decls += utils.write_dec("const int", \
                    "*m_size[" + str(len(classifier.m_size)) + \
                    "]") + '\n'

    for i in range(len(classifier.m_size)):
        if len(classifier.m_size[i]) == 0:
            inits += utils.write_attribution("m_size[" + str(i) + "]", \
                                             "NULL", tabs=1) + '\n'
            continue
        decls += utils.write_dec("const int", \
                    "tmp_size" + str(i) + "[" + \
                    str(len(classifier.m_size[i])) + "]", \
                    initValue=utils.toStr(classifier.m_size[i])) + '\n'
        inits += utils.write_attribution("m_size[" + str(i) + "]", \
                                         "tmp_size" + str(i), \
                                         tabs=1) + '\n'

    ### Initialize m_AttValues array
    decls += utils.write_dec("const " + decType, \
                    "**m_AttValues[" + str(len(classifier.m_AttValues)) + \
                    "]") + '\n'

    for i in range(len(classifier.m_AttValues)):
        if len(classifier.m_AttValues[i]) == 0:
            inits += utils.write_attribution("m_AttValues[" + str(i) + "]", \
                                             "NULL", tabs=1) + '\n'
            continue
        decls += utils.write_dec("const " + decType, \
                    "*tmp0_AttValues" + str(i) + "[" + \
                    str(len(classifier.m_AttValues[i])) + "]") + '\n'

        for j in range(len(classifier.m_AttValues[i])):
            decls += utils.write_dec("const " + decType, \
                    "tmp1_AttValues" + str(i) + "_" + str(j) + "[" + \
                    str(len(classifier.m_AttValues[i][j])) + "]", \
                    initValue=utils.toStr(classifier.m_AttValues[i][j])) + '\n'
            inits += utils.write_attribution("tmp0_AttValues" + str(i) + "[" + \
                                    str(j) + "]", \
                                    "tmp1_AttValues" + str(i) + "_" + str(j), \
                                    tabs=1) + '\n'
        
        inits += utils.write_attribution("m_AttValues[" + str(i) + "]", \
                                         "tmp0_AttValues" + str(i), \
                                         tabs=1) + '\n'

    ### Initialize m_b array
    decls += utils.write_dec("const " + decType, \
                    "m_b[" + str(len(classifier.m_b)) + "][" +\
                    str(len(classifier.m_b[0])) + "]", \
                    initValue=utils.toStr(classifier.m_b)) + '\n'
        
    inits += "}\n"

    # Definition of constant values
    defs += utils.write_define("INPUT_SIZE", str(len(classifier.attributes))) + \
    utils.write_define("NUM_CLASSES", str(len(classifier.classes))) + \
    utils.write_define("CLASS_INDEX", str(classifier.m_classIndex)) + \
    utils.write_define("M_EXPONENT", \
                       (utils.toFxp(classifier.m_exponent, opts) \
                        if opts['useFxp'] else \
                        str(classifier.m_exponent)))
        
    # Include of libraries
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs + inits)

def recover(model, opts):
    classifier = weka_SMO_PolyKernel()
    classifier.process(model, opts)

    return write_output(classifier, opts)

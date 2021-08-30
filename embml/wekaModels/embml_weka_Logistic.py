
from __future__ import print_function
from ..utils import utils

class weka_Logistic:

    def process(self, model, opts):

        self.m_Par = vars(model)['m_Par']
        self.m_NumPredictors = vars(model)['m_NumPredictors']
        self.m_classIndex = vars(model)['m_ClassIndex']
        self.m_numClasses = vars(model)['m_NumClasses']
        self.m_SelectedAttributes = vars(vars(vars(model)\
                            ['m_AttFilter'])['m_removeFilter'])\
                            ['m_SelectedAttributes']

        # Recover attribute names
        try:
            inputs = vars(vars(vars(vars(vars(vars(model)\
                        ['m_AttFilter'])['m_removeFilter'])\
                        ['m_InputRelAtts'])['m_Data'])\
                        ['m_Attributes'])['annotations'][1:]
        except:
            print ("Error: can't recover attribute names")
            exit(1)
        self.attributes = ['' for _ in range(len(inputs))]
        for i in range(len(inputs)):
            index = vars(inputs[i])['m_Index']
            self.attributes[index - (1 if index > self.m_classIndex else 0)] = vars(inputs[i])['m_Name']

        # Recover class names
        try:
            outputs = vars(vars(vars(vars(vars(vars(vars(vars(model)\
                    ['m_AttFilter'])['m_OutputRelAtts'])['m_Data'])\
                    ['m_Attributes'])['annotations'][self.m_classIndex + 1])\
                    ['m_AttributeInfo'])['m_Hashtable'])['annotations'][1:]
        except:
            try:
                outputs = vars(vars(vars(vars(vars(vars(model)\
                        ['m_structure'])['m_Attributes'])\
                        ['annotations'][self.m_classIndex + 1])\
                        ['m_AttributeInfo'])['m_Hashtable'])\
                        ['annotations'][1:]
            except:
                print ("Error: can't recover class names")
                exit(1)
        self.classes = ['' for _ in range(len(outputs) // 2)]
        for i in range(len(outputs)):
            if i % 2 == 0:
                continue
            self.classes[vars(outputs[i])['value']] = outputs[i - 1]

        new_Par = []
        for i in range(len(self.m_Par[0])):
            for j in range(len(self.m_Par)):
                new_Par.append(self.m_Par[j][i])
        self.m_Par = new_Par

        if opts['useFxp']:
            self.m_Par = [utils.toFxp(self.m_Par[_i], opts) \
                          for _i in range(len(self.m_Par))]

def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")
    
    # Classify function
    funcs += utils.write_output_classes(classifier.classes) + '\n' + \
    utils.write_func_init("int", "classify") + \
    utils.write_dec(decType, "prob[NUM_CLASSES]", tabs=1) + \
    utils.write_dec(decType, "newInstance[NUM_PREDICTORS + 1]", tabs=1)
    
    if opts['C']: 
        funcs += utils.write_dec("int", "i", tabs=1) + \
                utils.write_dec("int", "j", tabs=1)
    
    funcs += utils.write_attribution("newInstance[0]", \
                            (utils.toFxp(1.0, opts) \
                             if opts['useFxp'] else \
                             "1.0"), tabs=1) + \
    utils.write_for("i = 1", "i <= SELECTED_ATT_SIZE", "i++", tabs=1, inC=opts['C']) + \
    utils.write_if("m_SelectedAttributes[i] <= CLASS_INDEX", tabs=2) + \
    utils.write_attribution("newInstance[i]", \
                            "instance[m_SelectedAttributes[i - 1]]", \
                            tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_else(tabs=2) + \
    utils.write_attribution("newInstance[i]", \
                            "instance[m_SelectedAttributes[i]]", \
                            tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec(decType, \
                    "v[NUM_CLASSES]", \
                    initValue="{0}", \
                    tabs=1) + \
    utils.write_for("i = 0", "i < NUM_CLASSES - 1", "i++", tabs=1, inC=opts['C']) + \
    utils.write_for("j = 0", "j <= NUM_PREDICTORS", "j++", tabs=2, inC=opts['C']) + \
    utils.write_attribution("v[i]", \
                            ("fxp_sum(v[i], fxp_mul(m_Par[(i * (NUM_PREDICTORS + 1)) + j], newInstance[j]))" \
                            if opts['useFxp'] else \
                            "m_Par[(i * (NUM_PREDICTORS + 1)) + j] * newInstance[j]"), \
                            op=('' if opts['useFxp'] else '+'), tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_attribution("v[NUM_CLASSES - 1]", "0", tabs=1) + \
    utils.write_for("i = 0", "i < NUM_CLASSES", "i++", tabs=1, inC=opts['C']) + \
    utils.write_dec(decType, "acc", \
                    initValue=("0" if opts['useFxp'] else "0.0"), tabs=2) + \
    utils.write_for("j = 0", "j < NUM_CLASSES - 1", "j++", tabs=2, inC=opts['C']) + \
    utils.write_attribution("acc", \
                            ("fxp_sum(acc, fxp_exp(fxp_diff(v[j], v[i])))" \
                             if opts['useFxp'] else \
                            "expf(v[j] - v[i])"), \
                            op=('' if opts['useFxp'] else '+'), tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_attribution("prob[i]",
                            (("fxp_div(" +\
                              utils.toFxp(1.0, opts) + \
                              ", fxp_sum(acc, fxp_exp(-v[i])))")\
                             if opts['useFxp'] else \
                             "1.0 / (acc + expf(-v[i]))"), tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec("int", "indexMax", initValue="0", tabs=1) + \
    utils.write_for("i = 1", "i < NUM_CLASSES", "i++", tabs=1, inC=opts['C']) + \
    utils.write_if("prob[i] > prob[indexMax]", tabs=2) + \
    utils.write_attribution("indexMax", "i", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("indexMax", tabs=1) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_attributes(classifier.attributes) + '\n' + \
    utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec("const " + decType, \
                    "m_Par[" + str(len(classifier.m_Par)) + "]", \
                    initValue=utils.toStr(classifier.m_Par)) + '\n' + \
    utils.write_dec("const int", \
                    "m_SelectedAttributes[" + \
                    str(len(classifier.m_SelectedAttributes)) + "]", \
                    initValue=utils.toStr(classifier.m_SelectedAttributes))

    # Definition of constant values
    defs += utils.write_define("INPUT_SIZE", str(len(classifier.attributes) - 1)) + \
    utils.write_define("NUM_CLASSES", str(classifier.m_numClasses)) + \
    utils.write_define("SELECTED_ATT_SIZE", \
                       str(len(classifier.m_SelectedAttributes) - 1)) + \
    utils.write_define("NUM_PREDICTORS", str(classifier.m_NumPredictors)) + \
    utils.write_define("CLASS_INDEX", str(classifier.m_classIndex))
    
        
    # Include of libraries
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_define("OVERFLOW_DETECT") + \
        utils.write_include("\"FixedNum.h\"")
    else:
        incls += utils.write_include("<math.h>")

    return (incls + defs + decls + funcs)

def recover(model, opts):
    classifier = weka_Logistic()
    classifier.process(model, opts)

    return write_output(classifier, opts)

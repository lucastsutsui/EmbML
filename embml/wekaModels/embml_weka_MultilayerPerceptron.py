
from ..utils import utils
import sys

# Need to be adjusted when the model is too large
sys.setrecursionlimit(10000)

class weka_MultilayerPerceptron:

    def __init__(self):
        self.ids = dict()

    def bfsModel(self, m_inputs):

        queue = [(node, 0) for node in m_inputs]

        while len(queue) > 0:
            node, depth = queue[0]
            queue.pop(0)
        
            if not 'JavaObject' in str(type(node)) or\
               vars(node)['m_id'] in self.ids:
                continue

            if len(self.layers) <= depth:
                self.layers.append([])
                
                if depth > 0:
                    self.weights.append([])
                    self.biases.append([])

            currentIndex = len(self.ids)
            self.layers[depth].append(currentIndex)
            self.ids[vars(node)['m_id']] = currentIndex

            if depth > 0:
                m_inputList = vars(node)['m_inputList']
                m_weights = vars(node)['m_weights']
                newWeights = [0 for _i in range(len(self.layers[depth - 1]))]

                self.biases[depth - 1] += [m_weights[0]]

                for i in range(len(m_inputList)):
                    if not 'JavaObject' in str(type(m_inputList[i])):
                        continue
                    m_id = vars(m_inputList[i])['m_id']
                    idPrev =  self.ids[m_id] - min(self.layers[depth - 1])
                    newWeights[idPrev] = m_weights[i + 1]

                self.weights[depth - 1] += newWeights

            for i in vars(node)['m_outputList']:
                if 'JavaObject' in str(type(i)) and\
                   not 'NeuralEnd' in str(i):
                    queue.append((i, depth + 1))

    def process(self, model, opts):

        self.classIndex = vars(vars(model)['m_instances'])['m_ClassIndex']
        m_outputs = vars(model)['m_outputs']
        m_inputs = vars(model)['m_inputs']
        
        self.layers = []
        self.weights = []
        self.biases = []

        self.bfsModel(m_inputs)
        
        self.sizes = [len(_i) for _i in self.layers]

        self.m_attributeBases = vars(model)['m_attributeBases']
        self.m_attributeRanges = vars(model)['m_attributeRanges']

        self.m_attributeBases.pop(self.classIndex)
        self.m_attributeRanges.pop(self.classIndex)

        # Save classes and output values
        self.classes = [0 for _i in range(len(self.layers[-1]))]
        for node in m_outputs:
            connection = vars(node)['m_inputList'][0]
            idConnection = vars(connection)['m_id']
            self.classes[self.ids[idConnection] - min(self.layers[-1])] = vars(node)['m_id']

        # Save attributes and their order
        self.attributes = ['' for _i in range(len(self.layers[0]))]
        for node in m_inputs:
            m_id = vars(node)['m_id']
            self.attributes[self.ids[m_id]] = m_id

        if opts['useFxp']:
            self.m_attributeBases = [utils.toFxp(self.m_attributeBases[_i], opts) \
                                     for _i in range(len(self.m_attributeBases))]
            self.m_attributeRanges = [utils.toFxp(self.m_attributeRanges[_i], opts) \
                                    for _i in range(len(self.m_attributeRanges))]
            self.m_weights = [utils.toFxp(self.m_weights[_i], opts) \
                              for _i in range(len(self.m_weights))]
            if opts['pwl']:
                opts['pwlPoints'] = [utils.toFxp(opts['pwlPoints'][_i], opts) \
                                     for _i in range(len(opts['pwlPoints']))]
                opts['pwlCoefs'] = [[utils.toFxp(opts['pwlCoefs'][_i][_j], opts) \
                                for _j in range(len(opts['pwlCoefs'][_i]))]\
                                for _i in range(len(opts['pwlCoefs']))]
            
def sigFunction(opts):
    # Piecewise linear approximation
    if opts['pwl']:
        # First point
        pwlCode = utils.write_if("value < " + \
                    (opts['pwlPoints'][0] \
                     if opts['useFxp'] else \
                     ("%.10f" % opts['pwlPoints'][0])), tabs=1) + \
        utils.write_attribution("value", (utils.toFxp(0.0, opts) \
                                          if opts['useFxp'] else \
                                          "0.0"), tabs=2) + \
        utils.write_end(tabs=1)

        # Internal points
        for i in range(0, len(opts['pwlPoints']) - 1):
            pwlCode += utils.write_elseif("value < " + \
                                          (opts['pwlPoints'][i + 1] \
                                           if opts['useFxp'] else \
                                           ("%.10f" % opts['pwlPoints'][i + 1])), \
                                          tabs=1) + \
            utils.write_attribution("value", \
                    ("fxp_sum(fxp_mul(value, " + \
                     opts['pwlCoefs'][i][0] + "), " + \
                     opts['pwlCoefs'][i][1] + ")")\
                    if opts['useFxp'] else \
                    (("%.10f" % opts['pwlCoefs'][i][1]) + \
                     " + (" + ("%.10f" % opts['pwlCoefs'][i][0]) + \
                     " * value)"), tabs=2) + \
            utils.write_end(tabs=1)

        # Last point
        pwlCode += utils.write_else(tabs=1) + \
        utils.write_attribution("value", (utils.toFxp(1.0, opts) \
                                          if opts['useFxp'] else \
                                          "1.0"), tabs=2) + \
        utils.write_end(tabs=1)

        return pwlCode
    
    # Sigmoid Function
    sigCode = ""
    if opts['sigApprox']:
        sigCode = (("fxp_sum(" + utils.toFxp(0.5, opts) + \
                    ", fxp_mul(" + utils.toFxp(0.5, opts) + \
                    ", fxp_div(value, fxp_sum(" + \
                    utils.toFxp(1.0, opts) + ", my_abs(value)))))") \
                   if opts['useFxp'] else \
                   "0.5 * (value / (1.0 + my_abs(value))) + 0.5")
    else:
        sigCode = (("fxp_div(" + utils.toFxp(1.0, opts) + \
                    ", fxp_sum(" + utils.toFxp(1.0, opts) + \
                    ", fxp_exp(-value)))") if opts['useFxp'] else \
                   "1.0 / (1.0 + expf(-value))")

    return utils.write_if("value < " + (utils.toFxp(-45.0, opts) \
                                 if opts['useFxp'] else \
                                        "-45.0"), tabs=1) + \
    utils.write_attribution("value", (utils.toFxp(0.0, opts) \
                                      if opts['useFxp'] else \
                                      "0.0"), tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_elseif("value > " + (utils.toFxp(45.0, opts) \
                                     if opts['useFxp'] else \
                                     "45.0"), tabs=1) + \
    utils.write_attribution("value", (utils.toFxp(1.0, opts) \
                                      if opts['useFxp'] else \
                                      "1.0"), tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_else(tabs=1) + \
    utils.write_attribution("value", sigCode, tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("value", tabs=1)

def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'
    inits = "\nvoid initConnections(){\n"

    decType = ("FixedNum" if opts['useFxp'] else "float")

    # activation_hidden function
    funcs += utils.write_func_init(decType, "activation_function", \
                                   args=decType + " value") + \
    sigFunction(opts) + \
    utils.write_end(tabs=0)

    # calculateOutput function
    funcs += utils.write_func_init("inline void", "forward_pass", \
                                   args="const " + decType + " *input, " + \
                                   decType + " *output, " + \
                                   "const " + decType + " *coef, " + \
                                   "const " + decType + " *intercept, " + \
                                   "const int inputSize, const int outputSize") + \
    utils.write_dec("int", "i", initValue="0", tabs=1) + \
    utils.write_for("j = 0", "j < outputSize", "j++", tabs=1) + \
    utils.write_dec(decType, "acc", initValue=(utils.toFxp(0.0, opts) \
                                               if opts['useFxp'] else \
                                               "0.0"), tabs=2) + \
    utils.write_for("k = 0", "k < inputSize", "k++", tabs=2) + \
    utils.write_attribution("acc", \
                            "fxp_sum(acc, fxp_mul(coef[i++], input[k]))" \
                            if opts['useFxp'] else \
                            "(coef[i++] * input[k])", \
                            op=('' if opts['useFxp'] else '+'), \
                            tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_attribution("output[j]", \
                            "fxp_sum(acc, intercept[j])" \
                            if opts['useFxp'] else \
                            "acc + intercept[j]", \
                            tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_end(tabs=0) 
    
    # Classify function
    funcs += utils.write_output_classes(classifier.classes) + '\n' + \
    utils.write_func_init("int", "classify") + \
    utils.write_for("i = 0", "i < INPUT_SIZE", "i++", tabs=1) + \
    utils.write_if("m_attributeRanges[i] != " + ("0"\
                                                 if opts['useFxp'] else \
                                                 "0.0"), tabs=2) + \
    utils.write_attribution("instance[i]", \
            ("fxp_div(fxp_diff(instance[i], m_attributeBases[i]), m_attributeRanges[i])"\
             if opts['useFxp'] else \
             "(instance[i] - m_attributeBases[i]) / m_attributeRanges[i]"), \
                            tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_else(tabs=2) + \
    utils.write_attribution("instance[i]", \
                            ("fxp_diff(instance[i], m_attributeBases[i])"\
                            if opts['useFxp'] else \
                            "(instance[i] - m_attributeBases[i])"), \
                            tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec(decType, "*input", initValue="buffer1", tabs=1) + \
    utils.write_dec(decType, "*output", initValue="buffer2", tabs=1) + \
    utils.write_for("i = 0", "i < INPUT_SIZE", "i++", tabs=1) + \
    utils.write_attribution("input[i]", "instance[i]", tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_for("i = 0", "i < N_LAYERS - 1", "i++", tabs=1) + \
    utils.write_call("forward_pass(input, output, coefs[i], intercepts[i], sizes[i], sizes[i + 1])", tabs=2) + \
    utils.write_for("j = 0", "j < sizes[i + 1]", "j++", tabs=2) + \
    utils.write_attribution("output[j]", "activation_function(output[j])", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_if("(i + 1) != (N_LAYERS - 1)", tabs=2) + \
    utils.write_dec(decType, "*tmp", initValue="input", tabs=3) + \
    utils.write_attribution("input", "output", tabs=3) + \
    utils.write_attribution("output", "tmp", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec("int", "indMax", "0", tabs=1) + \
    utils.write_for("i = 0", \
                    "i < sizes[N_LAYERS - 1]", "i++", tabs=1) + \
    utils.write_if("output[i] > output[indMax]", tabs=2) + \
    utils.write_attribution("indMax", "i", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("indMax", tabs=1) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_attributes(classifier.attributes) + '\n' + \
    utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec(decType, "buffer1[" + str(max(classifier.sizes[::2])) + "]") + \
    utils.write_dec(decType, "buffer2[" + str(max(classifier.sizes[1::2])) + "]") + \
    utils.write_dec("const " + decType, \
                    "m_attributeBases[INPUT_SIZE]", \
                    initValue=utils.toStr(classifier.m_attributeBases)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "m_attributeRanges[INPUT_SIZE]", \
                    initValue=utils.toStr(classifier.m_attributeRanges)) + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(classifier.sizes), \
                    "sizes[N_LAYERS]", \
                    initValue=utils.toStr(classifier.sizes)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "*coefs[N_LAYERS - 1]") + '\n'

    ### Initialize coef array
    for i in range(len(classifier.weights)):
        if len(classifier.weights[i]) == 0:
            inits += utils.write_attribution("coefs[" + str(i) + "]", \
                                             "NULL", tabs=1)
            continue
        decls += utils.write_dec("const " + decType, \
                    "coefs_" + str(i) + "[" + \
                    str(len(classifier.weights[i])) + "]", \
                    initValue=utils.toStr(classifier.weights[i])) + '\n'
        inits += utils.write_attribution("coefs[" + str(i) + "]", \
                                         "coefs_" + str(i), \
                                         tabs=1)

    ### Initialize intercepts array
    decls += utils.write_dec("const " + decType, \
                    "*intercepts[N_LAYERS - 1]") + '\n'
    for i in range(len(classifier.biases)):
        if len(classifier.biases[i]) == 0:
            inits += utils.write_attribution("intercepts[" + str(i) + "]", \
                                             "NULL", tabs=1)
            continue
        decls += utils.write_dec("const " + decType, \
                    "intercepts_" + str(i) + "[" + \
                    str(len(classifier.biases[i])) + "]", \
                    initValue=utils.toStr(classifier.biases[i])) + '\n'
        inits += utils.write_attribution("intercepts[" + str(i) + "]", \
                                         "intercepts_" + str(i), \
                                         tabs=1)

    inits += "}\n"
    
    # Definition of constant values
    defs += utils.write_define("INPUT_SIZE", str(len(classifier.attributes))) + \
    utils.write_define("CLASS_INDEX", str(classifier.classIndex)) + \
    utils.write_define("N_LAYERS", str(len(classifier.layers))) + \
    utils.write_define("N_NEURONS", str(len(classifier.weights))) + \
    utils.write_define("my_abs(x)", "(((x) > (0.0)) ? (x) : -(x))") 
        
    # Include of libraries
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")
    else:
        incls += utils.write_include("<math.h>")

    return (incls + defs + decls + funcs + inits)

def recover(model, opts):
    classifier = weka_MultilayerPerceptron()
    classifier.process(model, opts)

    return write_output(classifier, opts)

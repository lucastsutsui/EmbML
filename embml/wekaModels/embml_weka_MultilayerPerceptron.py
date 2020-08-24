
from ..utils import utils
import sys

# Need to be adjusted when the model is too large
sys.setrecursionlimit(10000)

class weka_MultilayerPerceptron:

    def __init__(self):
        self.ids = dict()
        self.idsOutput = dict()
        self.idsInput = dict()
        self.sigIds = dict()

    def setIds(self, node, offset):
        if not 'JavaObject' in str(type(node)) or\
           'NeuralEnd' in str(node) or\
           vars(node)['m_id'] in self.ids:
            return

        for i in vars(node)['m_inputList']:
            if 'JavaObject' in str(type(i)):
                self.setIds(i, offset)
            
        self.ids[vars(node)['m_id']] = len(self.ids) + offset
    
        if 'SigmoidUnit' in str(vars(node)['m_methods']):
            self.sigIds[self.ids[vars(node)['m_id']]] = True

    # build the graph for the neural network
    def buildGraph(self, node, visited):
        if not 'JavaObject' in str(type(node)) or\
           'NeuralEnd' in str(node):
            return 

        m_inputList = vars(node)['m_inputList']
        m_id = self.ids[vars(node)['m_id']]
        m_weights = vars(node)['m_weights']

        if visited[m_id]:
            return
        visited[m_id] = True
        self.weights[m_id].append((-1, m_weights[0])) # -1 means bias

        for i in range(len(m_inputList)):
            if not 'JavaObject' in str(type(m_inputList[i])):
                continue

            if 'NeuralEnd' in str(m_inputList[i]):
                idNext = self.idsInput[vars(m_inputList[i])['m_id']]
            else:
                idNext = self.ids[vars(m_inputList[i])['m_id']]
            
            self.buildGraph(m_inputList[i], visited)
            self.weights[m_id].append((idNext, m_weights[i + 1]))

    def process(self, model, opts):

        self.classIndex = vars(vars(model)['m_instances'])['m_ClassIndex']
        m_outputs = vars(model)['m_outputs']
        m_inputs = vars(model)['m_inputs']

        # First, set ids for neurons in input layer
        for node in m_inputs:
            if not vars(node)['m_id'] in self.idsInput:
                self.idsInput[vars(node)['m_id']] = len(self.idsInput)

        # Second, set ids for neurons in hidden layers
        for node in m_outputs:
            for c in vars(node)['m_inputList']:
                self.setIds(c, len(self.idsInput))
      
        # Last, set ids for neurons in output layer
        for node in m_outputs:
            if not vars(node)['m_id'] in self.idsOutput:
                self.idsOutput[vars(node)['m_id']] = len(self.idsOutput) + \
                                                      len(self.ids) + \
                                                      len(self.idsInput)
        self.weights = [[] for _i in range(len(self.ids) + \
                                          len(self.idsOutput) + \
                                          len(self.idsInput))]
        visited = [False for _i in range(len(self.weights))]

        # Recover weights for each neuron
        for node in m_outputs:
            m_id = self.idsOutput[vars(node)['m_id']]
            self.weights[m_id].append((-1, 0)) # bias
    
            for c in vars(node)['m_inputList']:
                if not 'JavaObject' in str(type(c)):
                    continue
        
                self.buildGraph(c, visited)
                idNext = self.ids[vars(c)['m_id']]
                self.weights[m_id].append((idNext, 1))

        self.m_attributeBases = vars(model)['m_attributeBases']
        self.m_attributeRanges = vars(model)['m_attributeRanges']

        self.m_attributeBases.pop(self.classIndex)
        self.m_attributeRanges.pop(self.classIndex)

        # Save indices for sigmoid nodes
        self.sigmoids = []
        for i in range(len(self.weights)):
            if i in self.sigIds:
                self.sigmoids.append(True)
            else:
                self.sigmoids.append(False)

        # Save classes and output values
        self.first_output = min(self.idsOutput.values())
        self.classes = [0 for _i in range(len(self.idsOutput))]
        for i in self.idsOutput:
            self.classes[self.idsOutput[i] - self.first_output] = i

        # Save attributes and their order
        self.attributes = ['' for _i in range(len(self.idsInput))]
        for i in self.idsInput:
            self.attributes[self.idsInput[i]] = i

        # Initialize m_connections and m_weights arrays
        self.m_connections = []
        self.m_weights = []
        self.m_indicesMap = [0]
        
        for i in range(len(self.weights)):
            if len(self.weights[i]) == 0:
                # case in which neurons are connected to nothing (input neurons)
                self.m_indicesMap.append(self.m_indicesMap[-1])
                continue

            wei = []
            con = []
            for j,k in self.weights[i]:
                if j == -1:
                    # bias is the first number in array wei
                    wei.insert(0, k)
                    continue
                wei.append(k)
                con.append(j)
            
            self.m_indicesMap.append(len(con) + self.m_indicesMap[-1])
            self.m_connections += con
            self.m_weights += wei

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
                     ("%.10f" % opts['pwlPoints'][0])), tabs=3) + \
        utils.write_attribution("value", (utils.toFxp(0.0, opts) \
                                          if opts['useFxp'] else \
                                          "0.0"), tabs=4) + \
        utils.write_end(tabs=3)

        # Internal points
        for i in range(0, len(opts['pwlPoints']) - 1):
            pwlCode += utils.write_elseif("value < " + \
                                          (opts['pwlPoints'][i + 1] \
                                           if opts['useFxp'] else \
                                           ("%.10f" % opts['pwlPoints'][i + 1])), \
                                          tabs=3) + \
            utils.write_attribution("value", \
                    ("fxp_sum(fxp_mul(value, " + \
                     opts['pwlCoefs'][i][0] + "), " + \
                     opts['pwlCoefs'][i][1] + ")")\
                    if opts['useFxp'] else \
                    (("%.10f" % opts['pwlCoefs'][i][1]) + \
                     " + (" + ("%.10f" % opts['pwlCoefs'][i][0]) + \
                     " * value)"), tabs=4) + \
            utils.write_end(tabs=3)

        # Last point
        pwlCode += utils.write_else(tabs=3) + \
        utils.write_attribution("value", (utils.toFxp(1.0, opts) \
                                          if opts['useFxp'] else \
                                          "1.0"), tabs=4) + \
        utils.write_end(tabs=3)

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
                                 "-45.0"), tabs=3) + \
    utils.write_attribution("value", (utils.toFxp(0.0, opts) \
                                      if opts['useFxp'] else \
                                      "0.0"), tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_elseif("value > " + (utils.toFxp(45.0, opts) \
                                     if opts['useFxp'] else \
                                     "45.0"), tabs=3) + \
    utils.write_attribution("value", (utils.toFxp(1.0, opts) \
                                      if opts['useFxp'] else \
                                      "1.0"), tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_else(tabs=3) + \
    utils.write_attribution("value", sigCode, tabs=4) + \
    utils.write_end(tabs=3)

def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")

    # calculateOutput function
    funcs += "/* Function calculateOutput description:\n\
 * Returns the output value from a neuron\n\
 */\n" + \
    utils.write_func_init("inline void", "calculateOutput") + \
    utils.write_for("i = 0", "i < INPUT_SIZE", "i++", tabs=1) + \
    utils.write_attribution("m_value[i]", "instance[i]", tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_for("i = INPUT_SIZE", "i < NUMBER_OF_NEURONS", "i++", tabs=1) + \
    utils.write_dec(decType, "value", \
                    initValue="m_weights[m_indicesMap[i] + (i - INPUT_SIZE)]", \
                    tabs=2) + \
    utils.write_for("j = 0", \
                    "j < (m_indicesMap[i + 1] - m_indicesMap[i])", "j++", \
                    tabs=2) + \
    utils.write_attribution("value", \
                            ("fxp_sum(value, fxp_mul(m_weights[m_indicesMap[i] + (i - INPUT_SIZE) + j + 1], m_value[m_connections[m_indicesMap[i]] + j]))" \
                             if opts['useFxp'] else \
                             "m_weights[m_indicesMap[i] + (i - INPUT_SIZE) + j + 1] * m_value[m_connections[m_indicesMap[i]] + j]"), \
                            op=('' if opts['useFxp'] else '+'), tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_if("sigmoids[i]", tabs=2) + \
    sigFunction(opts) + \
    utils.write_end(tabs=2) + \
    utils.write_attribution("m_value[i]", "value", tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_end(tabs=0) 
    
    # Classify function
    funcs += utils.write_output_classes(classifier.classes) + '\n' + \
    utils.write_func_init("int", "classify") + \
    utils.write_dec(decType, "theArray[OUTPUT_SIZE]", tabs=1) + \
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
    utils.write_call("calculateOutput()", tabs=1) + \
    utils.write_for("i = 0", "i < OUTPUT_SIZE", "i++", tabs=1) + \
    utils.write_attribution("theArray[i]", \
                            "m_value[FIRST_OUTPUT + i]", \
                            tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec("int", "indexMax", initValue="0", tabs=1) + \
    utils.write_for("i = 1", "i < OUTPUT_SIZE", "i++", tabs=1) + \
    utils.write_if("theArray[i] > theArray[indexMax]", tabs=2) + \
    utils.write_attribution("indexMax", "i", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("indexMax", tabs=1) + \
    utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_attributes(classifier.attributes) + '\n' + \
    utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec("const " + decType, \
                    "m_attributeBases[INPUT_SIZE]", \
                    initValue=utils.toStr(classifier.m_attributeBases)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "m_attributeRanges[INPUT_SIZE]", \
                    initValue=utils.toStr(classifier.m_attributeRanges)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "m_weights[" + str(len(classifier.m_weights)) + ']', \
                    initValue=utils.toStr(classifier.m_weights)) + \
    utils.write_dec("const " + utils.chooseDataType(classifier.m_connections), \
                    "m_connections[" + str(len(classifier.m_connections)) + ']', \
                    initValue=utils.toStr(classifier.m_connections)) + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(classifier.m_indicesMap), \
                    "m_indicesMap[" + str(len(classifier.m_indicesMap)) + ']', \
                    initValue=utils.toStr(classifier.m_indicesMap)) + '\n' + \
    utils.write_dec("const bool", \
                    "sigmoids[NUMBER_OF_NEURONS]", \
                    initValue=utils.toStr(classifier.sigmoids).lower()) + '\n' + \
    utils.write_dec(decType, "m_value[NUMBER_OF_NEURONS]")

    # Definition of constant values
    defs += utils.write_define("INPUT_SIZE", str(len(classifier.attributes))) + \
    utils.write_define("CLASS_INDEX", str(classifier.classIndex)) + \
    utils.write_define("OUTPUT_SIZE", str(len(classifier.classes))) + \
    utils.write_define("FIRST_OUTPUT", str(classifier.first_output)) + \
    utils.write_define("NUMBER_OF_NEURONS", str(len(classifier.weights))) + \
    utils.write_define("my_abs(x)", "(((x) > (0.0)) ? (x) : -(x))") 
        
    # Include of libraries
    incls += utils.write_include("<Arduino.h>")
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")
    else:
        incls += utils.write_include("<math.h>")

    return (incls + defs + decls + funcs)

def recover(model, opts):
    classifier = weka_MultilayerPerceptron()
    classifier.process(model, opts)

    return write_output(classifier, opts)

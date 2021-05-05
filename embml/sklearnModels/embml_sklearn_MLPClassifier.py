
from __future__ import print_function
from ..utils import utils
import numpy as np

class sklearn_MLPClassifier:

    def process(self, model, opts):
        hidden_layer_sizes = model.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        self.n_layers = model.n_layers_
        self.coefs = model.coefs_
        self.intercepts = [_i.tolist() for _i in model.intercepts_]
        self.input_size = len(self.coefs[0])
        output_size = model.n_outputs_
        
        self.sizes = [self.input_size] + hidden_layer_sizes + [output_size]
            
        self.number_neurons = sum(self.sizes)
        self.classes = list(map(int, model.classes_.tolist()))

        self.activation_hidden = model.activation
        self.activation_output = model.out_activation_

        for i in range(len(self.coefs)):
            self.coefs[i] = self.coefs[i].transpose()
            self.coefs[i] = [_i.tolist() for _i in self.coefs[i]]
            for j in range(len(self.coefs[i])):
                for k in range(len(self.coefs[i][j])):
                    if abs(self.coefs[i][j][k]) < 1e-9:
                        self.coefs[i][j][k] = 0.0

        self.class_threshold = -1.0
        if len(list(self.classes)) == 2:
            self.class_threshold = (model._label_binarizer.pos_label + \
                               model._label_binarizer.neg_label) / 2.0
            
        if opts['useFxp']:
            self.class_threshold = utils.toFxp(\
                                    self.class_threshold, opts)
            self.coefs = [[[utils.toFxp(self.coefs[_i][_j][_k], opts)\
                        for _k in range(len(self.coefs[_i][_j]))]\
                        for _j in range(len(self.coefs[_i]))] \
                        for _i in range(len(self.coefs))]
            self.intercepts = [[utils.toFxp(self.intercepts[_i][_j], opts) \
                            for _j in range(len(self.intercepts[_i]))] \
                            for _i in range(len(self.intercepts))]
            if opts['pwl']:
                opts['pwlPoints'] = [utils.toFxp(opts['pwlPoints'][_i], opts) \
                                     for _i in range(len(opts['pwlPoints']))]
                opts['pwlCoefs'] = [[utils.toFxp(opts['pwlCoefs'][_i][_j], opts) \
                                for _j in range(len(opts['pwlCoefs'][_i]))]\
                                for _i in range(len(opts['pwlCoefs']))]

def activation_function(activation, opts):
    if activation == 'softmax':
        funcCode = ("fxp_diff(x, max_elem)" \
                    if opts['useFxp'] else \
                    "(x - max_elem)")
    elif activation == 'logistic':
        # Piecewise linear approximation
        if opts['pwl']:
            # First point
            pwlCode = utils.write_if("x < " + \
                                (opts['pwlPoints'][0] \
                                 if opts['useFxp'] else \
                                 ("%.10f" % opts['pwlPoints'][0])), tabs=1) + \
            utils.write_ret((utils.toFxp(0.0, opts) \
                             if opts['useFxp'] else \
                             "0.0"), tabs=2) + \
            utils.write_end(tabs=1)

            # Internal points
            for i in range(0, len(opts['pwlPoints']) - 1):
                pwlCode += utils.write_elseif("x < " + \
                                    (opts['pwlPoints'][i + 1] \
                                     if opts['useFxp'] else \
                                     ("%.10f" % opts['pwlPoints'][i + 1])), \
                                    tabs=1) + \
                utils.write_ret(("fxp_sum(fxp_mul(x, " + \
                        opts['pwlCoefs'][i][0] + "), " + \
                        opts['pwlCoefs'][i][1] + ")")\
                        if opts['useFxp'] else \
                        (("%.10f" % opts['pwlCoefs'][i][1]) + \
                         " + (" + ("%.10f" % opts['pwlCoefs'][i][0]) + \
                         " * x)"), tabs=2) + \
                utils.write_end(tabs=1)

            # Last point
            pwlCode += utils.write_else(tabs=1) + \
            utils.write_ret((utils.toFxp(1.0, opts) \
                             if opts['useFxp'] else \
                             "1.0"), tabs=2) + \
            utils.write_end(tabs=1)
            
            return pwlCode
        
        # Sigmoid Function
        if opts['sigApprox']:
            # Using approximation f(x) = x / (1 + abs(x))
            funcCode = (("fxp_sum(" + utils.toFxp(0.5, opts) + \
                         ", fxp_mul(" + utils.toFxp(0.5, opts) + \
                         ", fxp_div(x, fxp_sum(" + \
                         utils.toFxp(1.0, opts) + ", my_abs(x)))))") \
                        if opts['useFxp'] else \
                        "0.5 * (x / (1.0 + my_abs(x))) + 0.5")
        else:
            # Using original sigmoid function
            funcCode = (("fxp_div(" + utils.toFxp(1.0, opts) +\
                         ", fxp_sum(" + utils.toFxp(1.0, opts) +\
                         ", fxp_exp(-x)))")\
                        if opts['useFxp'] else \
                        "1.0 / (1.0 + expf(-x))")
    elif activation == 'relu':
        funcCode = (("max(" + utils.toFxp(0, opts) + ", x)")\
                    if opts['useFxp'] else \
                    "max(0.0, x)")
    else:
        print ("Activation function " + activation + " not supported!")
        exit(1)
        
    return utils.write_ret(funcCode, tabs=1)


def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'
    inits = "\nvoid initConnections(){\n"

    decType = ("FixedNum" if opts['useFxp'] else "float")

    # activation_hidden function
    funcs += utils.write_func_init(decType, "activation_hidden", \
                                   args=decType + " x") + \
    activation_function(classifier.activation_hidden, opts) + \
    utils.write_end(tabs=0)

    # activation_output function
    funcs += utils.write_func_init(decType, "activation_output", \
                                   args=decType + " x, " + decType + " max_elem") + \
    activation_function(classifier.activation_output, opts) + \
    utils.write_end(tabs=0)

    # forward_pass function
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
    funcs += utils.write_func_init("int", "classify") + \
    utils.write_dec(decType, "*input", initValue="buffer1", tabs=1) + \
    utils.write_dec(decType, "*output", initValue="buffer2", tabs=1) + \
    utils.write_for("i = 0", "i < INPUT_SIZE", "i++", tabs=1) + \
    utils.write_attribution("input[i]", "instance[i]", tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_for("i = 0", "i < N_LAYERS - 1", "i++", tabs=1) + \
    utils.write_call("forward_pass(input, output, coefs[i], intercepts[i], sizes[i], sizes[i + 1])", tabs=2) + \
    utils.write_if("(i + 1) != (N_LAYERS - 1)", tabs=2) + \
    utils.write_for("j = 0", "j < sizes[i + 1]", "j++", tabs=3) + \
    utils.write_attribution("output[j]", "activation_hidden(output[j])", tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_dec(decType, "*tmp", initValue="input", tabs=3) + \
    utils.write_attribution("input", "output", tabs=3) + \
    utils.write_attribution("output", "tmp", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_else(tabs=2) + \
    utils.write_dec(decType, "max_output", initValue="output[0]", tabs=3) + \
    utils.write_for("j = 1", "j < sizes[i + 1]", "j++", tabs=3) + \
    utils.write_if("output[j] > max_output", tabs=4) + \
    utils.write_attribution("max_output", "output[j]", tabs=5) + \
    utils.write_end(tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_for("j = 0", "j < sizes[i + 1]", "j++", tabs=3) + \
    utils.write_attribution("output[j]", "activation_output(output[j], max_output)", tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1)

    if len(list(classifier.classes)) == 2:
        funcs += utils.write_ret(\
            "classes[output[0] > CLASS_THRESHOLD]", tabs=1) + \
        utils.write_end(tabs=0)
    else:
        funcs += utils.write_dec("int", "indMax", "0", tabs=1) + \
        utils.write_for("i = 0", \
                    "i < sizes[N_LAYERS - 1]", "i++", tabs=1) + \
        utils.write_if("output[i] > output[indMax]", tabs=2) + \
        utils.write_attribution("indMax", "i", tabs=3) + \
        utils.write_end(tabs=2) + \
        utils.write_end(tabs=1) + \
        utils.write_ret("classes[indMax]", tabs=1) + \
        utils.write_end(tabs=0)
        
    # Declaration of global variables
    decls += utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec(decType, "buffer1[" + str(max(classifier.sizes[::2])) + "]") + \
    utils.write_dec(decType, "buffer2[" + str(max(classifier.sizes[1::2])) + "]") + \
    utils.write_dec("const " + utils.chooseDataType(classifier.sizes), \
                    "sizes[N_LAYERS]", \
                    initValue=utils.toStr(classifier.sizes)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "*coefs[N_LAYERS - 1]") + '\n'

    ### Initialize coef array
    for i in range(len(classifier.coefs)):
        if len(classifier.coefs[i]) == 0:
            inits += utils.write_attribution("coefs[" + str(i) + "]", \
                                             "NULL", tabs=1)
            continue
        decls += utils.write_dec("const " + decType, \
                    "coefs_" + str(i) + "[" + \
                    str(len(classifier.coefs[i])) + " * " + \
                    str(len(classifier.coefs[i][0])) + "]", \
                    initValue=utils.toStr1d(classifier.coefs[i])) + '\n'
        inits += utils.write_attribution("coefs[" + str(i) + "]", \
                                         "coefs_" + str(i), \
                                         tabs=1)
    
    ### Initialize intercepts array
    decls += utils.write_dec("const " + decType, \
                    "*intercepts[N_LAYERS - 1]") + '\n'
    for i in range(len(classifier.intercepts)):
        if len(classifier.intercepts[i]) == 0:
            inits += utils.write_attribution("intercepts[" + str(i) + "]", \
                                             "NULL", tabs=1)
            continue
        decls += utils.write_dec("const " + decType, \
                    "intercepts_" + str(i) + "[" + \
                    str(len(classifier.intercepts[i])) + "]", \
                    initValue=utils.toStr(classifier.intercepts[i])) + '\n'
        inits += utils.write_attribution("intercepts[" + str(i) + "]", \
                                         "intercepts_" + str(i), \
                                         tabs=1)

    # Declaration of classes array
    decls += utils.write_dec("const " + utils.chooseDataType(classifier.classes), \
                    "classes[NUM_CLASSES]", \
                    initValue=utils.toStr(classifier.classes)) + '\n'

    inits += "}\n"

    # Definition of constant values
    defs += utils.write_define("NUM_CLASSES", str(len(list(classifier.classes))))
    if len(list(classifier.classes)) == 2:
        defs += utils.write_define("CLASS_THRESHOLD", str(classifier.class_threshold))
    defs += utils.write_define("INPUT_SIZE", str(classifier.input_size)) + \
    utils.write_define("N_LAYERS", str(classifier.n_layers)) + \
    utils.write_define("N_NEURONS", str(classifier.number_neurons)) + \
    utils.write_define("my_abs(x)", "(((x) > (0.0)) ? (x) : -(x))") 
        
    # Include of libraries
    incls += utils.write_include("<Arduino.h>")
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs + inits)

def recover(model, opts):
    classifier = sklearn_MLPClassifier()
    classifier.process(model, opts)

    return write_output(classifier, opts)

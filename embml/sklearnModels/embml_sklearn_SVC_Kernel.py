
from __future__ import print_function
from ..utils import utils
import numpy as np

class sklearn_SVC_Kernel:

    def process(self, model, opts):
        self.gamma = model._gamma
        self.coef0 = model.coef0
        self.degree = model.degree
        self.model_l = model.support_.shape[0]
        self.nr_class = model.n_support_.shape[0]
        self.n_class = self.nr_class * (self.nr_class - 1) // 2
        self.input_size = model.support_vectors_.shape[1]
        self.kernel = model.kernel

        self.dual_coef = model._dual_coef_.tolist()
        self.support_vectors = model.support_vectors_.tolist()
        n_support = model.n_support_.tolist()
        self.intercept = model._intercept_.tolist()
        self.classes = list(map(int, model.classes_.tolist()))

        # Remove zeros from dual_coef matrix
        self.index_sv = []
        self.end = []
        lim = n_support
        for i in range(1, self.nr_class):
            lim[i] += lim[i-1]

        for i in range(len(self.dual_coef)):
            j = 0
            real_index = 0
            k = 0
            self.index_sv.append([])
            self.end.append([0])
            while real_index < self.model_l:
                if real_index >= lim[k]:
                    self.end[i].append(j)
                    k += 1
            
                if abs(self.dual_coef[i][j]) == 0.0:
                    self.dual_coef[i].pop(j)
                else:
                    self.index_sv[i].append(real_index)
                    j += 1
                real_index += 1
            self.end[i].append(j)
            
        if opts['useFxp']:
            self.gamma = utils.toFxp(self.gamma, opts)
            self.coef0 = utils.toFxp(self.coef0, opts)
            self.degree = utils.toFxp(self.degree, opts)
            
            self.dual_coef = [[utils.toFxp(self.dual_coef[_i][_j], opts) \
                            for _j in range(len(self.dual_coef[_i]))] \
                            for _i in range(len(self.dual_coef))]
            self.support_vectors = [[utils.toFxp(self.support_vectors[_i][_j], opts) \
                            for _j in range(len(self.support_vectors[_i]))] \
                            for _i in range(len(self.support_vectors))]
            self.intercept = [utils.toFxp(self.intercept[_i], opts) \
                            for _i in range(len(self.intercept))]

def kernelFunction(kernel, decType, opts):
    funcCode = utils.write_dec(decType, "sum", (utils.toFxp(0, opts)\
                                         if opts['useFxp'] else \
                                         "0.0"), tabs=1) + \
            utils.write_for("i = 0", "i < INPUT_SIZE", "i++", tabs=1)
            
    if kernel == 'linear':
        return funcCode + \
        utils.write_attribution("sum", ("fxp_sum(sum, fxp_mul(instance[i], y[i]))"\
                                          if opts['useFxp'] else \
                                          "instance[i] * y[i]"), \
                                 op=('' if opts['useFxp'] else '+'), tabs=2) + \
        utils.write_end(tabs=1) + \
        utils.write_ret("sum", tabs=1) + \
        utils.write_end(tabs=0)
    elif kernel == 'rbf':
        return funcCode + \
        utils.write_dec(decType, "tmp", ("fxp_diff(instance[i], y[i])" \
                                         if opts['useFxp'] else \
                                         "instance[i] - y[i]"), tabs=2) + \
        utils.write_attribution("sum", ("fxp_sum(sum, fxp_mul(tmp, tmp))" \
                                        if opts['useFxp'] else \
                                        "tmp * tmp"), \
                                 op=('' if opts['useFxp'] else '+'), tabs=2) + \
        utils.write_end(tabs=1) + \
        utils.write_ret(("fxp_exp(fxp_mul(-GAMMA, sum))" \
                         if opts['useFxp'] else \
                         "expf(-GAMMA * sum)"), tabs=1) + \
        utils.write_end(tabs=0)
    elif kernel == 'poly':
        return funcCode + \
        utils.write_attribution("sum", ("fxp_sum(sum, fxp_mul(instance[i], y[i]))" \
                                        if opts['useFxp'] else \
                                        "instance[i] * y[i]"), \
                                 op=('' if opts['useFxp'] else '+'), tabs=2) + \
        utils.write_end(tabs=1) + \
        utils.write_ret(("fxp_pow(fxp_sum(fxp_mul(GAMMA, sum), COEF0), DEGREE)" \
                         if opts['useFxp'] else \
                         "powf(GAMMA * sum + COEF0, DEGREE)"), tabs=1) + \
        utils.write_end(tabs=0)
    elif kernel == 'sigmoid':
        # TO DO: implement tanh function in fixed-point
        return funcCode + \
        utils.write_attribution("sum", ("fxp_sum(sum, fxp_mul(instance[i], y[i]))" \
                                        if opts['useFxp'] else \
                                        "instance[i] * y[i]"), \
                                 op=('' if opts['useFxp'] else '+'), tabs=2) + \
        utils.write_end(tabs=1) + \
        utils.write_ret(("// TO DO: implement tanh function in fixed-point" \
                         if opts['useFxp'] else \
                         "tanh(GAMMA * sum + COEF0)"), tabs=1) + \
        utils.write_end(tabs=0)
    else:
        print ("Kernel " + kernel + " not supported!")
        exit(1)


def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'
    inits = "\nvoid initConnections(){\n"

    decType = ("FixedNum" if opts['useFxp'] else "float")

    # Kernel Function
    funcs += utils.write_func_init(decType,
                                   "k_function",
                                   args="const " + decType + " *y") + \
    kernelFunction(classifier.kernel, decType, opts)
    
    # Classify function
    funcs += utils.write_func_init("int", "classify") + \
    utils.write_dec(decType, "k_value[MODEL_L]", tabs=1) + \
    utils.write_for("i = 0", "i < MODEL_L", "i++", tabs=1) + \
    utils.write_attribution("k_value[i]",
                            "k_function(support_vectors[i])",
                            tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec(utils.chooseDataType(classifier.n_class),
                    "vote[NR_CLASS]",
                    "{0}", tabs=1) + \
    utils.write_dec(utils.chooseDataType(classifier.n_class),
                    "p", "0", tabs=1) + \
    utils.write_for("i = 0", "i < NR_CLASS - 1", "i++", tabs=1) + \
    utils.write_for("j = i + 1", "j < NR_CLASS", "j++", tabs=2) + \
    utils.write_dec(decType, "sum", (utils.toFxp(0, opts)\
                                     if opts['useFxp'] else \
                                     "0.0"), tabs=3) + \
    utils.write_for("k = end[j - 1][i]", \
                    "k < end[j - 1][i + 1]", "k++", tabs=3) + \
    utils.write_attribution("sum",
        ("fxp_sum(sum, fxp_mul(dual_coef[j - 1][k], k_value[index_sv[j - 1][k]]))"\
         if opts['useFxp'] else\
         "dual_coef[j - 1][k] * k_value[index_sv[j - 1][k]]"),
        op=('' if opts['useFxp'] else '+'),
        tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_for("k = end[i][j]", \
                    "k < end[i][j + 1]", "k++", tabs=3) + \
    utils.write_attribution("sum",
        ("fxp_sum(sum, fxp_mul(dual_coef[i][k], k_value[index_sv[i][k]]))"\
         if opts['useFxp'] else\
         "dual_coef[i][k] * k_value[index_sv[i][k]]"),
        op=('' if opts['useFxp'] else '+'),
        tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_attribution("sum",
        ("fxp_sum(sum, intercept[p++])"\
         if opts['useFxp'] else\
         "intercept[p++]"),
        op=('' if opts['useFxp'] else '+'),
        tabs=3) + \
    utils.write_if("sum > 0", tabs=3) + \
    utils.write_inc_dec("vote[i]++", tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_else(tabs=3) + \
    utils.write_inc_dec("vote[j]++", tabs=4) + \
    utils.write_end(tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_dec("int", "indMax", "0", tabs=1) + \
    utils.write_for("i = 1", "i < NR_CLASS", "i++", tabs=1) + \
    utils.write_if("vote[i] > vote[indMax]", tabs=2) + \
    utils.write_attribution("indMax", "i", tabs=3) + \
    utils.write_end(tabs=2) + \
    utils.write_end(tabs=1) + \
    utils.write_ret("classes[indMax]", tabs=1) + \
    utils.write_end(tabs=0)
        
    # Declaration of global variables
    decls += utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n'

    ### Initialize dual_coef array
    decls += utils.write_dec("const " + decType, \
                    "*dual_coef[NR_CLASS - 1]") + '\n'
    for i in range(len(classifier.dual_coef)):
        if len(classifier.dual_coef[i]) == 0:
            inits += utils.write_attribution("dual_coef[" + str(i) + "]", \
                                             "NULL", tabs=1)
            continue
        decls += utils.write_dec("const " + decType, \
                    "dual_coef_" + str(i) + "[" + \
                    str(len(classifier.dual_coef[i])) + "]", \
                    initValue=utils.toStr(classifier.dual_coef[i])) + '\n'
        inits += utils.write_attribution("dual_coef[" + str(i) + "]", \
                                         "dual_coef_" + str(i), \
                                         tabs=1)

    ### Initialize support_vectors array
    decls += utils.write_dec("const " + decType, \
                    "support_vectors[MODEL_L][INPUT_SIZE]",
                    utils.toStr(classifier.support_vectors)) + '\n'

    ### Initialize end array
    decls += utils.write_dec("const " + utils.chooseDataType(classifier.end), \
                    "end[NR_CLASS - 1][NR_CLASS + 1]",
                    utils.toStr(classifier.end)) + '\n'

    ### Initialize index_sv array
    decls += utils.write_dec("const " + utils.chooseDataType(classifier.index_sv), \
                    "*index_sv[NR_CLASS - 1]") + '\n'
    for i in range(len(classifier.index_sv)):
        if len(classifier.index_sv[i]) == 0:
            inits += utils.write_attribution("index_sv[" + str(i) + "]", \
                                             "NULL", tabs=1)
            continue
        decls += utils.write_dec("const " + utils.chooseDataType(classifier.index_sv), \
                    "index_sv_" + str(i) + "[" + \
                    str(len(classifier.index_sv[i])) + "]", \
                    initValue=utils.toStr(classifier.index_sv[i])) + '\n'
        inits += utils.write_attribution("index_sv[" + str(i) + "]", \
                                         "index_sv_" + str(i), \
                                         tabs=1)
    
    ### Initialize intercept array
    decls += utils.write_dec("const " + decType, \
                    "intercept[N_CLASS]",
                    initValue=utils.toStr1d(classifier.intercept)) + '\n'

    # Declaration of classes array
    decls += utils.write_dec("const " + utils.chooseDataType(classifier.classes), \
                    "classes[NUM_CLASSES]", \
                    initValue=utils.toStr(classifier.classes)) + '\n'

    inits += "}\n"

    # Definition of constant values
    defs += utils.write_define("NUM_CLASSES", str(len(classifier.classes))) + \
    utils.write_define("INPUT_SIZE", str(classifier.input_size)) + \
    utils.write_define("GAMMA", (classifier.gamma if opts['useFxp'] else ("%.10f" % classifier.gamma))) + \
    utils.write_define("COEF0", str(classifier.coef0)) + \
    utils.write_define("DEGREE", str(classifier.degree)) + \
    utils.write_define("N_CLASS", str(classifier.n_class)) + \
    utils.write_define("NR_CLASS", str(classifier.nr_class)) + \
    utils.write_define("MODEL_L", str(classifier.model_l))
        
    # Include of libraries
    incls += utils.write_include("<Arduino.h>")
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs + inits)

def recover(model, opts):
    classifier = sklearn_SVC_Kernel()
    classifier.process(model, opts)

    return write_output(classifier, opts)

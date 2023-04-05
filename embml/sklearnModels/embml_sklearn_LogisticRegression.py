
from ..utils import utils

class sklearn_LogisticRegression:

    def process(self, model, opts):
        self.coef = model.coef_.tolist()
        self.intercept = model.intercept_.tolist()
        self.classes = list(map(int, model.classes_.tolist()))
        self.input_size = len(self.coef[0])
            
        if opts['useFxp']:
            self.coef = [[utils.toFxp(self.coef[_i][_j], opts) \
                          for _j in range(len(self.coef[_i]))] \
                         for _i in range(len(self.coef))]
            self.intercept = [utils.toFxp(_i, opts)\
                              for _i in self.intercept]
            
        if len(self.classes) == 2:
            self.coef = self.coef[0]
            self.intercept = self.intercept[0]

def write_output(classifier, opts):
    funcs = '\n'
    decls = '\n'
    defs = '\n'
    incls = '\n'

    decType = ("FixedNum" if opts['useFxp'] else "float")
    
    # Classify function
    funcs += utils.write_func_init("int", "classify")

    if len(classifier.classes) == 2:
        funcs += utils.write_dec(decType, "score", tabs=1)
    else:
        funcs += utils.write_dec("int", "indMax", initValue="0", tabs=1) + \
            utils.write_dec(decType, "scores[NUM_CLASSES]", tabs=1)

    if opts['C']: 
        funcs += utils.write_dec("int", "i", tabs=1) + \
                utils.write_dec("int", "j", tabs=1)

    if len(classifier.classes) == 2:
        funcs += utils.write_attribution("score", "intercept", tabs=1) + \
            utils.write_for("i = 0", "i < INPUT_SIZE", "i++", tabs=1, inC=opts['C']) + \
            utils.write_attribution("score", \
                                    ("fxp_sum(score, fxp_mul(coef[i], instance[i]))" \
                                     if opts['useFxp'] else \
                                     "(coef[i] * instance[i])"), \
                                    op=('' if opts['useFxp'] else '+'), tabs=2) + \
            utils.write_end(tabs=1) + \
            utils.write_ret("classes[((score > 0) ? 1 : 0)]", tabs=1) + \
            utils.write_end(tabs=0)
    else:
        funcs += utils.write_for("i = 0", "i < NUM_CLASSES", "i++", tabs=1, inC=opts['C']) + \
            utils.write_attribution("scores[i]", "intercept[i]", tabs=2) + \
            utils.write_for("j = 0", "j < INPUT_SIZE", "j++", tabs=2, inC=opts['C']) + \
            utils.write_attribution("scores[i]", \
                                    ("fxp_sum(scores[i], fxp_mul(coef[i][j], instance[j]))" \
                                     if opts['useFxp'] else \
                                     "(coef[i][j] * instance[j])"), \
                                    op=('' if opts['useFxp'] else '+'), tabs=3) + \
            utils.write_end(tabs=2) + \
            utils.write_if("scores[i] " + \
                           ("<" if len(classifier.classes) == 2 else ">") + \
                           " scores[indMax]", tabs=2) + \
            utils.write_attribution("indMax", "i", tabs=3) + \
            utils.write_end(tabs=2) + \
            utils.write_end(tabs=1) + \
            utils.write_ret("classes[indMax]", tabs=1) + \
            utils.write_end(tabs=0)

    # Declaration of global variables
    decls += utils.write_dec(decType, "instance[INPUT_SIZE + 1]") + '\n' + \
    utils.write_dec("const " + decType, \
                    ("coef[INPUT_SIZE]"
                     if len(classifier.classes) == 2 else
                     "coef[NUM_CLASSES][INPUT_SIZE]"), \
                    initValue=utils.toStr(classifier.coef)) + '\n' + \
    utils.write_dec("const " + decType, \
                    "intercept" + ("[NUM_CLASSES]" if len(classifier.classes) != 2 else ""), \
                    initValue=utils.toStr(classifier.intercept)) + '\n' + \
    utils.write_dec("const " + utils.chooseDataType(classifier.classes), \
                    "classes[NUM_CLASSES]", \
                    initValue=utils.toStr(classifier.classes)) + '\n'

    # Definition of constant values
    defs += utils.write_define("NUM_CLASSES", str(len(classifier.classes))) + \
    utils.write_define("INPUT_SIZE", str(classifier.input_size))
        
    # Include of libraries
    if opts['useFxp']:
        incls += utils.write_define("TOTAL_BITS", str(opts['totalBits'])) + \
        utils.write_define("FIXED_FBITS", str(opts['fracBits'])) + \
        utils.write_define("SIGNED") + \
        utils.write_include("\"FixedNum.h\"")

    return (incls + defs + decls + funcs)

def recover(model, opts):
    classifier = sklearn_LogisticRegression()
    classifier.process(model, opts)

    return write_output(classifier, opts)

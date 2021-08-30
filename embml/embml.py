
from __future__ import print_function
from . embmlWeka import recoverWeka
from . embmlSklearn import recoverSklearn
import javaobj
import pickle
import sys

def processOptions(opt):
    opts = dict()
    
    opts['useFxp'] = ('-fxp' in opt.split())
    if opts['useFxp']:
        if len(opt.split()) <= opt.split().index('-fxp') + 2:
            print ("Error: define numbers of integer and fractional bits")
            exit(1)

        opts['fracBits'] = int(opt.split()[opt.split().index('-fxp') + 2])
        opts['totalBits'] = int(opt.split()[opt.split().index('-fxp') + 1]) + opts['fracBits'] + 1

        if opts['totalBits'] != 8 and\
           opts['totalBits'] != 16 and\
            opts['totalBits'] != 32:
            print ("Error: <integer bits> + <fractional bits> needs to be equals to 7, 15, or 31")
            exit(1)

    opts['rules'] = ('-rules' in opt.split())
    
    opts['C'] = ('-c' in opt.split())

    opts['sigApprox'] = ('-sigApprox' in opt.split())
    opts['pwl'] = ('-sigPwl' in opt.split())
    if opts['pwl']:
        opts['nPoints'] = int(opt.split()[int(opt.split().index('-sigPwl')) + 1])
        if opts['nPoints'] == 2:
            # Use 2 points
            opts['pwlPoints'] = [-2.60060859307396, 2.60060859307396]
            opts['pwlCoefs'] = [[0.19226268856129256, 0.5]]
        else:
            # Use 4 points
            opts['pwlPoints'] = [-3.96049288887045136676, -1.6379627182375, 1.6379627182375, 3.96049288887045136676]
            opts['pwlCoefs'] = [[0.0588394235821312, 0.23303311868226695], [0.2218265772854816, 0.5], [0.0588394235821312, 0.766966881317733]]
    
    return opts
    
def wekaModel(inputFileName, outputFileName, opts=''):
    modelFile = open(inputFileName, "rb")
    marshaller = javaobj.JavaObjectUnmarshaller(modelFile)
    model = marshaller.readObject()
    opts = processOptions(opts)
        
    with open(outputFileName, "w") as output:
        output.write(recoverWeka(model, opts))

def sklearnModel(inputFileName, outputFileName, opts=''):
    modelFile = open(inputFileName, "rb")

    # Check Python version
    if sys.version_info[0] == 2:
        model = pickle.load(modelFile)
    elif sys.version_info[0] == 3:
        model = pickle.load(modelFile, encoding='latin1')
    else:
        # Default option
        model = pickle.load(modelFile, encoding='latin1')
    
    opts = processOptions(opts)

    with open(outputFileName, "w") as output:
        output.write(recoverSklearn(model, opts))

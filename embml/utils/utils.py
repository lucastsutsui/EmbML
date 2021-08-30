
import math

EPS = 1e-9

def write_dec(dataType, varName, initValue="", tabs=0):
    return ('\t' * tabs) + \
        dataType + \
        ' ' + \
        varName + \
        ((' = ' + initValue) if initValue != '' else '') + ';\n'

def write_end(tabs=0):
    return ('\t' * tabs) + '}\n'

def write_while(condition, tabs=0):
    return ('\t' * tabs) + 'while (' + condition + '){\n'

def write_for(var, condition, inc, tabs=0, inC=False):
    if inC:
        return ('\t' * tabs) + 'for (' + var + \
            '; ' + condition + '; ' + inc + '){\n'
    else:
        return ('\t' * tabs) + 'for (int ' + var + \
            '; ' + condition + '; ' + inc + '){\n'

def write_if(condition, tabs=0):
    return ('\t' * tabs) + 'if (' + condition + '){\n'

def write_elseif(condition, tabs=0):
    return ('\t' * tabs) + 'else if (' + condition + '){\n'

def write_else(tabs=0):
    return ('\t' * tabs) + 'else {\n'

def write_call(call, tabs=0):
    return ('\t' * tabs) + call + ';\n'

def write_inc_dec(var, tabs=0):
    return ('\t' * tabs) + var + ';\n'

def write_func_init(dataType, funcName, args=''):
    return dataType + ' ' + funcName + '(' + args + '){\n'

def write_attribution(firstVar, secondVar, op='', tabs=0):
    return ('\t' * tabs) + firstVar + ' ' + op + '= ' + secondVar + ';\n'

def write_ret(value, tabs=0):
    return ('\t' * tabs) + 'return ' + value + ';\n'

def write_define(varName, value=''):
    return '#define ' + varName + ' ' + value + '\n'

def write_include(libName):
    return '#include ' + libName + '\n'

def write_output_classes(classes):
    ret = "\n/* Function classify description:\n\
 * Instance array must be initializated, with appropriated attributes, before calling this function\n"

    for i in range(len(classes)):
        ret += " * Output number " +\
                 str(i) +\
                 " means that the instance was classified as " +\
                 classes[i] +\
                 "\n"
    
    ret += " */\n"
    return ret

def write_attributes(attributes):
    ret = "/* Instance array must be global\n\
 * Attributes MUST be sorted in instance array in the following order:\n"
    for i in attributes:
        ret += " * " +\
                 i +\
                 "\n"
        
    ret += " */\n"
    return ret

# Convert a float point number to its fixed point representation
# returns a string that has this representation in hexadecimal.
def toFxp(x, opts):
    if math.isnan(x):
        return "INF_POS"
    if opts['totalBits'] == 32:
        return "(FixedNum)0x%.8x" % (int(round(EPS + (x * (1 << opts['fracBits'])))) & 0xFFFFFFFF)
    if opts['totalBits'] == 16:
        return "(FixedNum)0x%.4x" % (int(round(EPS + (x * (1 << opts['fracBits'])))) & 0xFFFF)
    if opts['totalBits'] == 8:
        return "(FixedNum)0x%.2x" % (int(round(EPS + (x * (1 << opts['fracBits'])))) & 0xFF)

def to1d(v):
    try:
        v = list(v)
    except:
        return [v]
    
    if len(v):
        v1d = []
        for elem in v:
            v1d += to1d(elem)
        return v1d
    
    return v

def chooseDataType(v):
    v = to1d(v)

    if len(v) == 0 or (max(v) <= 127 and min(v) >= -128): return "int8_t"
    if max(v) <= 255 and min(v) >= 0: return "uint8_t"
    if max(v) <= 32767 and min(v) >= -32768: return "int16_t"
    if max(v) <= 65535 and min(v) >= 0: return "uint16_t"
    if max(v) <= 2147483647 and min(v) >= -2147483648: return "int32_t"
    if max(v) <= 4294967295 and min(v) >= 0: return "uint32_t"
    if max(v) <= 9223372036854775807 and min(v) >= -9223372036854775808: return "int64_t"
    if max(v) <= 18446744073709551615 and min(v) >= 0: return "uint64_t"
    else: return "int64_t"

def toStr(v):
    return str(v).replace('[','{').replace(']','}').replace('\'','').replace('nan','NAN')

def toStr1d(v):
    return '{' + str(v).replace('[','').replace(']','').replace('\'','') + '}'

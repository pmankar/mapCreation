import time
import sys

# global configuration variables
verboseMode = True

# global variables
initTime = time.time()


def write(stmnt, sType = "INFO", override = False):
    '''
    writing current status
    input:
    stmnt: statement
    sType: statement type INFO|ERR|WARN|OthersTags
    override: override default verboseMode
    '''
    global verboseMode
    if (override):
        writeS(stmnt, sType)
        return
    if (verboseMode):
        writeS(stmnt, sType)


def writeS(stmnt, sType = "INFO"):
    '''
    main writing; avoid using this, use write instead
    '''
    global initTime
    delay = time.time() - initTime
    # use sys.stdout.write for threadsafe printing
    sys.stdout.write( sType + "\t+" + "{0:.3f}".format(delay) + "\t" + stmnt + "\t" + "\n")
    # to give offset from previous print statement
    initTime = time.time()
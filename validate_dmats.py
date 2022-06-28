from CERGenerator import *
import sys

fin = sys.argv[1]
strict = int(sys.argv[2])
validate_dmats(fin,strict)

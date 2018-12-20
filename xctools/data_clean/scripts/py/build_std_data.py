from sparseData import LOAD_DATA,WRITE_DATA
import sys

X,Y = LOAD_DATA(sys.argv[1])
WRITE_DATA(X,Y,sys.argv[2],int(sys.argv[3]))
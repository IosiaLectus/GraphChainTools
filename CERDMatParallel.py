import sys
from CERGenerator import *

def main():
    metric_name = sys.argv[1]
    graphString1 = sys.argv[2]
    graphString2 = sys.argv[2]

    metric_from_name = {"minDistanceCUDA": minDistanceCUDA, "meanDistanceCUDA": meanDistanceCUDA, "specDistance": specDistance, "edgeCountDistance": edgeCountDistance, "disagreementCount": disagreementCount, "doublyStochasticMatrixDistance": doublyStochasticMatrixDistance}
    metric = metric_from_name[metric_name]

    g1 = stringToGraphList(graphString1)
    g2 = stringToGraphList(graphString2)

    d = metric(g1,g2)
    print(d)

# Do stuff
if __name__ == '__main__':
    main()

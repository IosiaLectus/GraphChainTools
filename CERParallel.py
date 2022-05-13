import sys
from CERGenerator import *

def main():
    ngraphs = int(sys.argv[1])
    metric_name = sys.argv[2]
    folder = sys.argv[3]

    metric_from_name = {"minDistanceCUDA": minDistanceCUDA, "meanDistanceCUDA": meanDistanceCUDA, "specDistance": specDistance, "edgeCountDistance": edgeCountDistance, "disagreementCount": disagreementCount, "doublyStochasticMatrixDistance": doublyStochasticMatrixDistance}
    metric = metric_from_name[metric_name]

    pairwise_distances_from_folder(folder, ngraphs, metric, True, metric_name)

# Do stuff
if __name__ == '__main__':
    main()

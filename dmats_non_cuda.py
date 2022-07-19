from CERGenerator import *
import sys


nshots = 200
p = float(sys.argv[1])
outfile = sys.argv[3]
q = float(sys.argv[2])

nvertices = 12
ngraphs = 100

metrics = [edgeCountDistance, disagreementCount, specDistance, minDistanceCUDA, meanDistanceCUDA, doublyStochasticMatrixDistance]
metricNames = {minDistanceCUDA: "minDistanceCUDA", meanDistanceCUDA: "meanDistanceCUDA", specDistance: "specDistance", edgeCountDistance: "edgeCountDistance", disagreementCount: "disagreementCount", doublyStochasticMatrixDistance: "doublyStochasticMatrixDistance"}

for m in metrics:
    mname = metricNames[m]
    print(f"doing metric = {mname}")
    chains_to_dmats_json_partial('data.json',outfile,m,mname,nvertices,p,q)

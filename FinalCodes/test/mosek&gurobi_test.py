import os
from math import sqrt

imageclass_list = ['WhiteNoise', 'GRFrough', 'GRFmoderate', 'GRFsmooth', 'LogGRF',
                   'LogitGRF', 'CauchyDensity', 'Shapes', 'ClassicImages', 'MicroscopyImages']
f = open("test/mosek&gurobi_test.txt", 'w+')

for data in ['DOTmark', 'random', 'Caffa', 'ellip']:
    for method in ['primal', 'dual', 'barrier']:
        if data == 'DOTmark':
            for imageclass in imageclass_list:
                print(data, method, imageclass, 'gurobi')
                print(data, method, imageclass, 'gurobi', file=f)
                s = os.popen("python solver_gurobi.py " + "--data " + data
                             + " --method " + method + " --image-class " + imageclass).read()
                print(s, file=f)

        else:
            for n in [16, sqrt(512), 32, sqrt(2048)]:
                print(data, method, int(n ** 2))

                print(data, method, int(n ** 2), 'gurobi', file=f)
                s = os.popen("python solver_gurobi.py" + " --data " + data
                             + " --method " + method + " --n " + str(n)).read()
                print(s, file=f)

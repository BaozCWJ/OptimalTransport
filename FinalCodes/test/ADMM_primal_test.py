import os
from math import sqrt
import time

imageclass_list = ['WhiteNoise', 'GRFrough', 'GRFmoderate', 'GRFsmooth', 'LogGRF',
                   'LogitGRF', 'CauchyDensity', 'Shapes', 'ClassicImages', 'MicroscopyImages']

f = open("test/ADMM_primal_test.txt", 'w+')

for data in ['DOTmark', 'random', 'Caffa', 'ellip']:
# for data in ['random', 'Caffa', 'ellip']:
# for data in ['DOTmark']:
    if data == 'DOTmark':
        for imageclass in imageclass_list:
            print(data, imageclass)

            print(data, imageclass, 'primal', file=f)
            start = time.time()
            s = os.popen("python ADMM_primal.py " + "--data " + data
                         + " --image-class " + imageclass).read()
            print('once time=', time.time() - start)
            print(s, file=f)

            # print(data, imageclass, 'dual', file=f)
            # s = os.popen("python ADMM_dual.py " + "--data " + data
            #              + " --image-class " + imageclass).read()
            # print(s, file=f)

    else:
        for n in [16, sqrt(512), 32, sqrt(2048)]:
            print(data, int(n**2))

            print(data, int(n**2), 'primal', file=f)
            start = time.time()
            s = os.popen("python ADMM_primal.py" + " --data " + data
                         + " --n " + str(n)).read()
            print('once time=', time.time() - start)
            print(s, file=f)

            # print(data, int(n ** 2), 'dual', file=f)
            # s = os.popen("python ADMM_dual.py" + " --data " + data
            #              + " --n " + str(n)).read()
            # print(s, file=f)

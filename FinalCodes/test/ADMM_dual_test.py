import os
from math import sqrt
import time

imageclass_list = ['WhiteNoise', 'GRFrough', 'GRFmoderate', 'GRFsmooth', 'LogGRF',
                   'LogitGRF', 'CauchyDensity', 'Shapes', 'ClassicImages', 'MicroscopyImages']
eps_dict = {''}
f = open("test/ADMM_dual_test.txt", 'w+')

for data in ['DOTmark', 'random', 'Caffa', 'ellip']:
# for data in ['random', 'Caffa', 'ellip']:
# for data in ['DOTmark']:
    if data == 'DOTmark':
        for imageclass in imageclass_list:
            print(data, imageclass)

            print(data, imageclass, 'dual', file=f)
            start = time.time()
            s = os.popen("python ADMM_dual.py " + "--data " + data
                         + " --image-class " + imageclass).read()
            print('once time=', time.time() - start)
            print(s, file=f)

    else:
        for n in [16, sqrt(512), 32, sqrt(2048)]:
            print(data, int(n**2))

            print(data, int(n**2), 'dual', file=f)
            start = time.time()
            s = os.popen("python ADMM_dual.py" + " --data " + data
                         + " --n " + str(n)).read()
            print('once time=', time.time() - start)
            print(s, file=f)

            '''
            random  n 16        : alpha 15, rho 11.5 或 9
                    n sqrt(512) : alpha 
            '''

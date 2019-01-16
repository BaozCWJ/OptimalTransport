import os
from math import sqrt, log
import time

imageclass_list = ['WhiteNoise', 'GRFrough', 'GRFmoderate', 'GRFsmooth', 'LogGRF',
                   'LogitGRF', 'CauchyDensity', 'Shapes', 'ClassicImages', 'MicroscopyImages']
loss_dict_dotmark = {'WhiteNoise': 6e-4, 'GRFrough': 1.4e-3, 'GRFmoderate': 3.9e-3, 'GRFsmooth': 2e-2,
                     'LogGRF': 1.8e-2, 'LogitGRF': 1.6e-2, 'CauchyDensity': 1.7e-2, 'Shapes': 2.3e-2,
                     'ClassicImages': 6.1e-2, 'MicroscopyImages': 1e-2}
loss_dict_notdot = {'random': 4.15e-2, 'ellip': 4.1, 'Caffa': 4.158}

f = open("test/sinkhorn_newton_notdot_test.txt", 'w+')

# for data in ['DOTmark', 'random', 'Caffa', 'ellip']:
for data in ['random', 'Caffa', 'ellip']:
    if data == 'DOTmark':
        n = 32
        for imageclass in imageclass_list:
            print(data, imageclass, 'sinkhorn_newton')

            print(data, imageclass, 'sinkhorn_newton', file=f)
            start = time.time()
            s = os.popen("python sinkhorn_newton.py " + "--data " + data
                         + " --image-class " + imageclass
                         + " --eps " + str(loss_dict_dotmark[imageclass] / log(n ** 4))).read()
            print(s)
            print('once time=', time.time() - start)
            print(s, file=f)

    else:
        for n in [16, sqrt(512), 32, sqrt(2048)]:
            print(data, int(n ** 2), 'sinkhorn_newton')

            print(data, int(n ** 2), 'sinkhorn_newton', file=f)
            start = time.time()
            s = os.popen("python sinkhorn_newton.py" + " --data " + data
                         + " --n " + str(n) + " --eps " +
                         str(loss_dict_notdot[data] / log(n ** 4))
                         ).read()
            print(s)
            print('once time=', time.time() - start)
            print(s, file=f)

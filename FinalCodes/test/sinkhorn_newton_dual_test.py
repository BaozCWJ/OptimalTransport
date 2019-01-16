import os
from math import sqrt, log
import time

imageclass_list = ['WhiteNoise', 'GRFrough', 'GRFmoderate', 'GRFsmooth', 'LogGRF',
                   'LogitGRF', 'CauchyDensity', 'Shapes', 'ClassicImages', 'MicroscopyImages']
loss_dict_dotmark = {'WhiteNoise': 6e-4, 'GRFrough': 1.4e-3, 'GRFmoderate': 3.9e-3, 'GRFsmooth': 2e-2,
                     'LogGRF': 1.8e-2, 'LogitGRF': 1.6e-2, 'CauchyDensity': 1.7e-2, 'Shapes': 2.3e-2,
                     'ClassicImages': 6.1e-2, 'MicroscopyImages': 1e-2}
loss_dict_notdot = {'random': 4.15e-4, 'ellip': 4.1, 'Caffa': 4.158}

# f = open("test/sinkhorn_newton_dual_dotmark_test.txt", 'w+')

eps_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2, 3, 4, 5, 6, 7]
for eps in eps_list[::-1]:
    print("sinkhorn_newton_dual ellip eps=" + str(eps))
    s = os.popen("python sinkhorn_newton_dual.py " + "--data " + "ellip"
                 + " --n 16 "
                 + " --eps " + str(eps)
                 ).read()
    print(s)

# for eps in [5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 40, 50]:
#     print("sinkhorn_newton_dual DOTmark eps=" + str(eps))
#     s = os.popen("python sinkhorn_newton_dual.py " + "--data " + "DOTmark"
#                  + " --image-class " + "Shapes"
#                  + " --eps " + str(eps)
#                  ).read()
#     print(s)

# for data in ['DOTmark', 'random', 'Caffa', 'ellip']:
# for data in ['random', 'Caffa', 'ellip']:
# for data in ['DOTmark']:
#     if data == 'DOTmark':
#         n = 32
#         for imageclass in imageclass_list:
#             print(data, imageclass)
#
#             print(data, imageclass, 'sinkhorn_newton_dual', file=f)
#             start = time.time()
#             s = os.popen("python sinkhorn_newton_dual.py " + "--data " + data
#                          + " --image-class " + imageclass
#                          # + " --eps " + str(loss_dict_dotmark[imageclass] / log(n ** 4))
#                          ).read()
#             print(s)
#             print('once time=', time.time() - start)
#             print(s, file=f)
#
#     else:
#         for n in [16, sqrt(512), 32, sqrt(2048)]:
#             print(data, int(n ** 2))
#
#             print(data, int(n ** 2), 'sinkhorn_newton_dual', file=f)
#             start = time.time()
#             s = os.popen("python sinkhorn_newton_dual.py" + " --data " + data
#                          + " --n " + str(n)
#                          # + " --eps " + str(loss_dict_notdot[data] / log(n ** 4))
#                          ).read()
#             print(s)
#             print('once time=', time.time() - start)
#             print(s, file=f)

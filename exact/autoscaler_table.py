import json, autoscaler as A
A.lam0, A.U, A.L = 1.2, 8, 1
res = [A.analyse(a) for a in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0, 100.0)]
json.dump(res, open('autoscaler_exact.json', 'w'), indent=1)

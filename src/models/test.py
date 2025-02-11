import numpy as np
from options.european import European
from options.digital import Digital
from options.asian import Asian


k = 4
ts = np.array([1, 2, 3, 4, 10])

ec = European(k)
payoff = ec.calculate_payoff(ts)
print(payoff)  

di = Digital(k)
payoff = di.calculate_payoff(ts)
print(payoff)  

a = Asian(k)
payoff = a.calculate_payoff(ts)
print(payoff)  
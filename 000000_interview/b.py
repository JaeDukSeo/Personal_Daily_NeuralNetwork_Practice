import numpy as np

def np_sig(x): return 1/(1 + np.exp(-1 * x))
data = np.array([0.2,0.4,0.5,0.5,0.4])
print(np_sig(data))
print('----------------------')





# --- end code --
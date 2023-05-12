import numpy as np

# Ontology Layer
M = np.asarray([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

save_dir = 'data/'

M = M.T
num_lower, num_upper = M.shape

correlation_method_1 = np.zeros((num_lower, num_lower), dtype=np.int64)
# print(correlation_method_1.shape)
for i in range(num_lower):
    for j in range(num_lower):
        if np.all(M[i] == M[j]):
            correlation_method_1[i, j] = 1
    # print(correlation_method_1[i].tolist())
    
correlation_method_1 = np.zeros((num_lower, num_lower), dtype=np.int64)
    
np.save(save_dir + 'ontology_correlation_method_1.npy', correlation_method_1)


num_all = num_lower + num_upper
correlation_method_2 = np.zeros((num_all, num_all), dtype=np.int64)

M_pad = np.pad(M.T, ((0, 0), (num_upper, 0)), 'constant', constant_values=0)
# print(M_pad.shape)
correlation_method_2[:num_upper] = M_pad
correlation_method_2 = correlation_method_2 + correlation_method_2.T
# for i in range(50):
#     print(correlation_method_2[i].tolist())

np.save(save_dir + 'ontology_correlation_method_2.npy', correlation_method_2)

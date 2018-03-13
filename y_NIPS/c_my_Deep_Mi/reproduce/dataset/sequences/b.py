import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt

np.random.seed(7689)
tf.set_random_seed(678)


# 
human_pos  = open("human_pos.fa", "r")
human_pos_list = human_pos.readlines()
human_pos_list = list(map(lambda s: s.strip(), human_pos_list))

print(len(human_pos_list) )
print(human_pos_list[1])
print(human_pos_list[-1],'\n')

human_pos_list_final = human_pos_list[1::2]

print(len(human_pos_list_final))
print(human_pos_list_final[0])
print(human_pos_list_final[-1],'\n')

human_neg  = open("human_neg.fa", "r")
human_neg_list = human_neg.readlines()[:1726]
human_neg_list = list(map(lambda s: s.strip(), human_neg_list))

print(len(human_neg_list) )
print(human_neg_list[1])
print(human_neg_list[-1],'\n')

human_neg_list_final = human_neg_list[1::2]

print(len(human_neg_list_final))
print(human_neg_list_final[0])
print(human_neg_list_final[-1],'\n')

# 
print("==================")
for x in human_neg_list_final:
    print(x)
    print(len(x))



# == end code ==
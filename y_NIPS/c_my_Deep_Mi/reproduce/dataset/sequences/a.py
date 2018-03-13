import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt


human_pos  = open("human_pos.fa", "r")
human_pos_list = human_pos.readlines() 

print(len(human_pos_list) )
print(human_pos_list[1])
print(human_pos_list[-1])

human_pos_list_final = human_pos_list[1::2]

print(len(human_pos_list_final))
print(human_pos_list_final[0])
print(human_pos_list_final[-1])


human_neg  = open("human_neg.fa", "r")
human_neg_list = human_neg.readlines()[:1726]

print(len(human_neg_list) )
print(human_neg_list[1])
print(human_neg_list[-1])

human_neg_list_final = human_neg_list[1::2]

print(len(human_neg_list_final))
print(human_neg_list_final[0])
print(human_neg_list_final[-1])


# == end code ==
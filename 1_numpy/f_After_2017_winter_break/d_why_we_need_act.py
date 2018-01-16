import numpy as np
import numpy

# NOTE: On online python this does work for some reason.
np.random.seed(1234)

def identity_act(x):
      return x
  
def d_identity_act(x):
  return 1


x = np.array([
    [1,2,1],
    [3,4,1],
    [5,6,1],
    [7,8,1]
])

y = np.array([
    [8],
    [16],
    [24],
    [32]
])

num_epoch =150
lr = 0.0001

w1 = np.random.randn(3,4)
w2 = np.random.randn(4,10)
w3 = np.random.randn(10,1)


for iter in range(num_epoch):
    
    layer_1 = x.dot(w1)
    layer_1_act = identity_act(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = identity_act(layer_2)
    
    layer_3 = layer_2_act.dot(w3)
    layer_3_act = identity_act(layer_3)

    cost = np.square(layer_3_act - y).sum() * 0.5
    print("Current Iter: ",iter, " current Cost: ",cost)

    grad_3_part_1= layer_3_act - y
    grad_3_part_2= d_identity_act(layer_3)
    grad_3_part_3=layer_2_act
    grad_3 =   grad_3_part_3.T.dot(grad_3_part_1*grad_3_part_2)
    
    grad_2_part_1= (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2= d_identity_act(layer_2)
    grad_2_part_3=layer_1_act
    grad_2 =   grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2) 

    grad_1_part_1= (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2= d_identity_act(layer_1)
    grad_1_part_3= x
    grad_1 =     grad_1_part_3.T.dot(grad_1_part_1*grad_1_part_2)   

    w1 = w1 - lr*grad_1
    w2 = w2 - lr*grad_2
    w3 = w3 - lr*grad_3

print("----------------")    
print("After 100 Iter Result")
layer_1 = x.dot(w1)
layer_1_act = identity_act(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = identity_act(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = identity_act(layer_3)
print(layer_3_act)

print("----------------")    
print("Results Rounded: ")
print(np.round(layer_3_act))

print("----------------")    
print("Ground Truth: ")
print(y)

print("----------------")    
print("One Linear Equation ")
print("k = w1.dot(w2.dot(w3))")
k = w1.dot(w2.dot(w3))
one_liner = x.dot(k)
print(one_liner)

print("----------------")    
print("One Linear Equation Rounded: ")
print(np.round(one_liner))



# print("----------------")    
# print("Calculated Weigths: ")
# print(w1)
# print(w2)
# print(w3)






# -- end code --
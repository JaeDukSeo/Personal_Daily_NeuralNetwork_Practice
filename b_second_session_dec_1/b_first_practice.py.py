import tensorflow as tf
import numpy as np



np.random.seed(1234)
tf.set_random_seed(1234)


x_data = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y_data= np.array([
    [0],
    [0],
    [0],
    [1]
    
])

print(x_data.shape)
print(y_data.shape)

# 1. Make Graph
graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder('float',[4,3])
    y = tf.placeholder('float',[4,1])

    
    w1 = tf.Variable(tf.random_normal([3,4],stddev=0.001))
    w2 = tf.Variable(tf.random_normal([4,1],stddev=0.001))

    layer_1 = tf.matmul(x,w1)    
    layer_1_act = tf.nn.sigmoid(layer_1)

    layer_2 = tf.matmul(layer_1_act,w2)
    layer_2_act = tf.nn.sigmoid(layer_2)

    cost = tf.square(layer_2_act - y)
    optimize = tf.train.AdamOptimizer(0.1).minimize(cost)
    

# 2. Make session
sess = tf.Session(graph=graph)

with sess:
    
    sess.run(tf.global_variables_initializer())
    for iter in range(100):
        
        optimzes = sess.run([cost,optimize],feed_dict={x:x_data,y:y_data})
        print(type(optimzes))
        print(optimzes[0])
    
    print('-----------------------------------------')
    optimze = sess.run(layer_2_act,feed_dict={x:x_data,y:y_data})
    print(optimze)











# ------ end ------
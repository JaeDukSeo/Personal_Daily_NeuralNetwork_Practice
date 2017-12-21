import tensorflow as tf
import numpy as np



# 1. Make the Graph
graph = tf.Graph()
with graph.as_default():

    input_1 = tf.placeholder('float',[3,3])
    batch_norm = tf.contrib.layers.batch_norm(input_1)

    input_2 = tf.placeholder('float',[1,4,4,1])
    max_pool = tf.nn.max_pool(input_2,ksize=[1,2, 2,1], strides=[1, 2, 2,1], padding='SAME')
    avg_pool = tf.nn.avg_pool(input_2,ksize=[1,2, 2,1], strides=[1, 2, 2,1], padding='SAME')
    

# 2. Make the Session 
with tf.Session(graph = graph) as sess : 
    
    sess.run(tf.global_variables_initializer())


    batch_norm_input = np.array([
        [3,3,3],
        [3,4,3],
        [3,3,3]
    ])

    batch_norm_data = sess.run(batch_norm, feed_dict={input_1:batch_norm_input})
    print batch_norm_data,'\n\n'

    max_pool_input = np.array([
        [3,3,0.4,2],
        [3,3,0.4,1],
        [3,3,3,4],
        [3,3,3,4],
    ])
    max_pool_input = np.expand_dims(max_pool_input,axis=0)
    max_pool_input = np.expand_dims(max_pool_input,axis=3)
    
    # Max Pool - choose the max element
    max_pool_data = sess.run(max_pool, feed_dict={input_2:max_pool_input})
    print max_pool_data,'\n\n'

    # Avg Pool - Combine the values and average them
    avg_pool_data = sess.run(avg_pool, feed_dict={input_2:max_pool_input})
    print avg_pool_data




# ------ END OF THE CODE ---
import tensorflow as tf


# 1. Preprocess data
# 2. Make the graph and the variable 
# 3. Make the session and calculate

# 1. Skip 

# 2. Graph 
graph = tf.Graph()
with graph.as_default():
    a = tf.placeholder("float") # Create a symbolic variable 'a'
    b = tf.placeholder("float") # Create a symbolic variable 'b'
    y = tf.multiply(a, b) # multiply the symbolic variables

# 3. Session
with tf.Session(graph = graph) as sess:
    # you need to initialize all variables
    sess.run(tf.global_variables_initializer())
    temp  = sess.run(y,feed_dict={a: 5, b: 3})
    print temp
    print type(temp)







#  -------- END OF THE CODE ---------
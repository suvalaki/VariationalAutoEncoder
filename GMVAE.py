import tensorflow as tf
import tensorflow_probability as tfp

#solve for posteriors of latent (hidden) parameters. q(z|x) = RV(h(x); sigma)
def inference_model(inputs, encoding_dim, 
                    categorical_dim, latent_dim):
    
    # condirional model for posterior inference of y
    with tf.variable_scope("y"):
        fcl1 = tf.layers.dense(inputs, encoding_dim, activation=tf.nn.relu, name='layer1')

        # define the categorical distribution
        y_logit = tf.layers.dense(fcl1, categorical_dim, activation=None, name='logit')
        y_p = tf.nn.softmax(y_logit, name='prob')
        y = tfp.edward2.Categorical( probs=y_p ,name='val')
    
    # conditional model for posterior inference of z|y
    with tf.variable_scope("z_gy"):
        fcl1 = tf.layers.dense(inputs, encoding_dim, activation=tf.nn.relu, name='layer1')

        zm,zv,zsd,z = ([None] * categorical_dim for i in range(4))
        for i in range(categorical_dim):
            zm[i] = tf.layers.dense(inputs, latent_dim, activation=None, name='m_{:d}'.format(i))
            zv[i] = tf.layers.dense(inputs, latent_dim, activation=tf.nn.softplus, name='v_{:d}'.format(i))
            zsd[i] = tf.sqrt(zv[i], name='sd_{:d}'.format(i))
            z[i] = tfp.edward2.Normal(loc=zm[i], scale=zsd[i], name='conditional_{:d}'.format(i))
        
    with tf.variable_scope("x"):
        z_mix = tfp.edward2.Mixture(cat=y, components=z, name='val')
        
    return y_logit, y_p, y, zm, zv, z, z_mix

#the process we observe. p(x|z) = RV(g(z); beta)
def observation_model(inputs, latent_dim, encoding_dim, input_dim):

    with tf.variable_scope("")
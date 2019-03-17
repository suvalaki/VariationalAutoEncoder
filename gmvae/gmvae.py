import tensorflow as tf
import numpy as np
from math import floor
from scipy.stats import mode

from tensorflow.nn import softmax_cross_entropy_with_logits_v2 as cross_entropy_with_logits

def gaussian_sample(mean, var, scope=None):
    #Permit sampling from a gaussin distribution
    with tf.variable_scope(scope, 'gaussian_sample'):
        sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
        sample.set_shape(mean.get_shape())
        return sample

def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    # Clipped 
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = tf.clip_by_value(logits, -max_val, max_val,
                                name='clipped_logit')
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)

def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = tf.add(var, eps, name='clipped_var')
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis)

class GMVAE():

    def __init__(self, input_dim, kind='binary', components=1, encoding_dims = [500], 
        encoding_activations=[tf.nn.relu] , latent_dims = 64, random_state=None):

        self.kind = kind
        self.input_dim = input_dim
        self.components = components
        self.encoding_dims = encoding_dims
        self.latent_dims = latent_dims

        self.encoder_dimensions = self.encoding_dims
        self.decoder_dimensions = self.encoding_dims[::-1]

        self.encoding_activations = encoding_activations
        self.decoder_activations = encoding_activations[::-1]
        
        self.random_state = random_state
        
        self.loss_history = []
        self.ent_history = []


    @staticmethod
    def _qy_graph(x, k, embedding_dims, embedding_activations):
        reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qy')) > 0
        # -- q(y)
        with tf.variable_scope('qy'):

            for i, (dims, activation) in enumerate(zip(embedding_dims, embedding_activations)):
                x = tf.layers.dense(inputs=x, units=dims, name=f'layer_{i}', activation=activation, reuse=reuse)

            qy_logit = tf.layers.dense(inputs=x, units=k, name='logit', reuse=reuse)
            qy = tf.nn.softmax(qy_logit, name='prob')

        return qy_logit, qy

    @staticmethod
    def _qz_graph(x, y, embedding_dims, embedding_activations, latent_dims):
        input_dim = x.shape[-1]
        reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qz')) > 0
        # -- q(z)
        with tf.variable_scope('qz'):
            xy = tf.concat(values=(x, y), axis=1, name='xy/concat')

            for i, (dims, activation) in enumerate(zip(embedding_dims, embedding_activations)):
                xy = tf.layers.dense(inputs=xy, units=dims, name=f'layer_{i}', activation=activation, reuse=reuse)

            zm = tf.layers.dense(inputs=xy, units=latent_dims, name='zm', activation=None, reuse=reuse)
            zv = tf.layers.dense(inputs=xy, units=latent_dims, name='zv', activation=tf.nn.softplus, reuse=reuse)
            z = gaussian_sample(zm, zv, 'z')
            
        return z, zm, zv

    @staticmethod
    def _labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
        xy_loss = -log_bernoulli_with_logits(x, px_logit)
        xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
        return xy_loss - np.log(0.1)

    @staticmethod
    def _labeled_loss_numeric(x, px_logit, z, zm, zv, zm_prior, zv_prior):
        xy_loss = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=px_logit)
        xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
        return xy_loss - np.log(0.1)

    @staticmethod
    def _px_graph(z, y, embedding_dims, embedding_activations, output_dims):
        """
        Builds the decoder graph for x|zk
        z: tf.tensor(shape=(?,latent_dims))
        embedding_dims: list(int) : list of layer dimensions
        """
        latent_dims = z.shape[-1]
        reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='px')) > 0
        # -- p(z)
        with tf.variable_scope('pz'):
            zm = tf.layers.dense(inputs=y, units=latent_dims, name='zm', activation=None, reuse=reuse)
            zv = tf.layers.dense(inputs=y, units=latent_dims, name='zv', activation=tf.nn.softplus, reuse=reuse)
        # -- p(x)
        with tf.variable_scope('px'):
            for i, (dims, activation) in enumerate(zip(embedding_dims, embedding_activations)):
                z = tf.layers.dense(inputs=z, units=dims, name=f'layer_{i}', activation=activation, reuse=reuse)

            px_logit = tf.layers.dense(inputs=z, units=output_dims, name='logit', activation=None, reuse=reuse)
        return zm, zv, px_logit


    def build(self):

        input_dim=self.input_dim 
        kind=self.kind 
        random_state=self.random_state

        # Setup Graph
        tf.reset_default_graph()

        # set random seed
        if random_state is not None:
            tf.random.set_random_seed(random_state)


        # create dataset iterator...?
            
        # Input Placeholder
        x = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name='x')

        # binarize data and create a y "placeholder"
        if kind == 'binary':
            with tf.name_scope('x_binarized'):
                xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
        elif kind == 'regression':
            xb = x
        with tf.name_scope('y_'):
            y_ = tf.cast(x=tf.fill(dims=tf.stack([tf.shape(x)[0], self.components]), value=0.0), dtype=tf.float32)

        # propose distribution over y
        qy_logit, qy = self._qy_graph(xb, self.components, self.encoder_dimensions, self.encoding_activations)

        # for each proposed y, infer z and reconstruct x
        z, zm, zv, zm_prior, zv_prior, px_logit = [[None] * self.components for i in range(6)]
        zm_joint = [None] * self.components
        for i in range(self.components):
            with tf.name_scope('graphs/hot_at{:d}'.format(i)):
                y = tf.add(y_, tf.constant(value=np.eye(self.components)[i], dtype=tf.float32, name='hot_at_{:d}'.format(i)))
                z[i], zm[i], zv[i] = self._qz_graph(xb, y, self.encoder_dimensions, self.encoding_activations, self.latent_dims)
                zm_prior[i], zv_prior[i], px_logit[i] = self._px_graph(z[i], y, self.decoder_dimensions, self.decoder_activations, self.input_dim)
                # additional mixture metric
                zm_joint[i] = qy[:,i,None]*zm[i]

        with tf.name_scope('loss'):
            with tf.name_scope('neg_entropy'):
                nent = -cross_entropy_with_logits(logits=qy_logit, labels=qy)
            losses = [None] * self.components
            for i in range(self.components):
                with tf.name_scope('loss_at{:d}'.format(i)):
                    if kind=='binary':
                        losses[i] = self._labeled_loss(xb, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
                    elif kind=='regression':
                        losses[i] = self._labeled_loss_numeric(xb, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
            with tf.name_scope('final_loss'):
                loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in range(self.components)])
                
        # Additional Scope to examine Latent Space
        zm_latent = tf.add_n(zm_joint)

        # compile the model 
        train_step = tf.train.AdamOptimizer().minimize(loss)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        sess_info = (sess, qy_logit, nent, loss, train_step)

        # Add variables to the class
        self.var_qy_logit = qy_logit
        self.var_qy = qy

        self.var_z = z
        self.var_zm = zm
        self.var_zv = zv
        self.var_zm_prior = zm_prior 
        self.var_zv_prior = zv_prior
        self.var_px_logit = px_logit
        self.var_zm_joint = zm_joint
        self.zm_latent = zm_latent

        self.var_nent = nent
        self.var_losses = losses
        self.var_loss = loss

        self.train_step = train_step
        self.sess = sess
        self.sess_info = sess_info

        return self

    @staticmethod
    def iterator_results_run(iterator_dict, session, target, agg_func):

        # initialise the itterator_dict
        # Note iterators should be same length
        next_element_dict = {}                                 
        for key in iterator_dict.keys():
            iterator = iterator_dict[key]
            iterator = iterator.make_one_shot_iterator()
            next_element_dict[key] = iterator.get_next()

        # Run over all the data
        result = []
        while True:
            try:
                # Get the data as a dict
                iterator_values = {}
                for key in iterator_dict.keys():
                    iterator_values[key] = session.run(next_element_dict[key])[0]

                # Train over all data
                if agg_func is not None:
                    result.append(session.run(target, feed_dict=iterator_values))
                else:
                    session.run(target, feed_dict=iterator_values)
            except tf.errors.OutOfRangeError:
                break

        # Aggregate results
        if agg_func is not None:
            if type(target) == list: 
                result_output = []
                for i in range(len(target)):
                    result_output[i] = agg_func(result[i])
            else:
                result = agg_func(result)
            return result_output


        
    
    def validation_acc(self, X, labels):
        """
        Find the 
43
        X is an itterator
        """
        #logits = self.sess.run(self.var_qy_logit, feed_dict={'x:0': X})
        logits = self.iterator_results_run(
            iterator_dict = {'x:0': X}, 
            session = self.sess, 
            target = self.var_qy_logit, 
            agg_func = lambda x: x )
        )
        cat_pred = logits.argmax(1)
        real_pred = np.zeros_like(cat_pred)
        for cat in range(logits.shape[1]):
            idx = cat_pred == cat
            lab = labels[idx].argmax(1)
            if len(lab) == 0:
                continue
            real_pred[cat_pred == cat] = mode(lab).mode[0] # most common  value in array
        return np.mean(real_pred == labels.argmax(1))

    def fit(self, X_train, batch_size = 100, subsample=None, epochs=1, iterep=1, 
            y_train=None, X_test=None, y_test=None, subsample_test=None):
        """
        Fits trains the model for the given number of epochs and itterations
        """

        def xstr(s):
            if s is None:
                return 0.0
            else:
                return s


        # turn the dataset into a tensorflow.data.Dataset
        # https://www.tensorflow.org/guide/datasets#consuming_values_from_an_iterator
        dataset_X_train = tf.data.Dataset.from_tensor_slices(X_train)
        batched_dataset_X_train = dataset_X_train.batch(1000)
        iterator_X_train = batched_dataset_X_train.make_one_shot_iterator()
        next_element_X_train = iterator_X_train.get_next()

        if X_train is not None and y_train is not None:
            dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            batched_dataset_train = dataset_train.batch(batch_size)
            iterator_train = batched_dataset_train.make_one_shot_iterator()
            next_element_train = iterator_train.get_next()

        if X_test is not None:
            dataset_X_test = tf.data.Dataset.from_tensor_slices(X_test)
            batched_dataset_X_test = dataset_X_test.batch(batch_size)
            iterator_X_test = batched_dataset_X_test.make_one_shot_iterator()
            next_element_X_test = iterator_X_test.get_next()

        if X_test is not None and y_train is not None:
            dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            batched_dataset_test = dataset_test.batch(batch_size)
            iterator_test = batched_dataset_test.make_one_shot_iterator()
            next_element_test = iterator_test.get_next()

        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)) if y_train is not None else None
        dataset_X_test = tf.data.Dataset.from_tensor_slices(X_test) if X_test is not None else None
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test,y_test)) if (X_test is not None and y_test is not None) else None


        # Get session params
        (sess, qy_logit, nent, loss, train_step) = self.sess_info
        idxs = list(range(X_train.shape[0]))

        if X_test is not None:
            idxs_x_test = list(range(X_test.shape[0]))

        for i in range(iterep * epochs):

            #sample only a proprtion of the data
            if subsample is not None:
                if type(subsample) == float and subsample < 1:
                    #is a percentage
                    idx = np.random.choice(idxs, floor(len(idxs) * subsample), False)
                else:
                    idx = np.random.choice(idxs, subsample, False)
            else:
                idx = idxs

            if subsample_test is not None and X_test is not None:
                if type(subsample_test) == float and subsample_test < 1:
                    #is a percentage
                    idx_test = np.random.choice(idxs_x_test, floor(len(idxs_x_test) * subsample_test), False)
                else:
                    idx_test = np.random.choice(idxs_x_test, subsample_test, False)
            else:
                idx_test = idxs_x_test

            # run the training step

            # init the iterator because is destroyed after use
            self.iterator_results_run(
                iterator_dict = {'x:0': batched_dataset_train}, 
                session = sess, 
                target = train_step, 434343
                agg_func = None )

            print('this far')

            # print out metrics
            if (i + 1) %  iterep == 0:

                train_ent, train_loss, train_acc, test_ent, test_loss, test_acc = [None for i in range(6)] 

                #train_ent, train_loss = sess.run([self.var_nent, self.var_loss], feed_dict={'x:0': X_train[idx]})

                # caclulate entropy
                train_ent, train_loss = self.iterator_results_run(
                    iterator_dict = {'x:0': next_element_X_train}, 
                    session = sess, 
                    target = [self.var_nent, self.var_loss], 
                    agg_func = lambda x: np.sum(np.array(x), axis=0) )

                train_ent, train_loss = -train_ent.mean(), train_loss.mean()

                if y_train is not None:
                    #train_acc = self.validation_acc(X_train[idx], y_train[idx])
                    train_acc = self.validation_acc(X_train[idx], y_train[idx])
                else:
                    train_acc = 0.0

                if X_test is not None: 
                    test_ent, test_loss = sess.run([self.var_nent, self.var_loss], feed_dict={'x:0': X_test[idx_test]})
                    test_ent, test_loss = -test_ent.mean(), test_loss.mean()

                    if y_test is not None:
                        test_acc = self.validation_acc(X_test[idx_test], y_test[idx_test])
                    else:
                        test_acc = None

                else:
                    test_ent, test_loss, test_acc = 0.0 , 0.0, 0.0

                if i+1 == iterep:
                    string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                            .format('tr_ent', 'tr_loss', 'tr_acc', 't_ent', 't_loss', 't_acc', 'epoch'))
                    print(string)

                it = int(floor((i + 1) / iterep))
                string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                        .format(
                            train_ent, train_loss, train_acc, 
                            test_ent, test_loss, test_acc, 
                            it
                        ))
                print(string)

    def predict(self, att, X):
        return self.sess.run(att, feed_dict={'x:0':X})
    
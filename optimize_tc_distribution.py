import h5py
from tqdm import tqdm
import datetime
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

from data_processing import DataProcessing
from plotter import Plotter
from debug import debug_tensor_shape

def are_tensors_equal(t1, t2):
    assert t1.shape == t2.shape
    return tf.math.count_nonzero(tf.math.equal(t1,t2))==t1.shape
        
def tensorflow_assignment(tensor, mask, lambda_op):
    """
    Emulate assignment by creating a new tensor.
    The mask must be 1 where the assignment is intended, 0 everywhere else.
    """
    assert tensor.shape == mask.shape
    other = lambda_op(tensor)
    return tensor * (1 - mask) + other * mask

def tensorflow_wasserstein_1d_loss(indata, outdata):
    """Calculates the 1D earth-mover's distance."""
    loss = tf.cumsum(indata) - tf.cumsum(outdata)
    #loss = tf.abs( loss )
    loss = tf.math.square( loss )

    # throw away all differences smaller than `constant`
    # constant = 1e-1
    # mask = tf.greater(loss, constant * tf.ones_like(loss))
    # loss = tf.multiply(loss, tf.cast(mask, dtype=tf.float32))
    
    loss = tf.reduce_sum( loss )
    return loss

# https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
class Architecture(tf.keras.Model):
    """Neural network model definition."""
    def __init__(self, inshape, kernel_size):
        super().__init__()
        assert len(inshape)==1

        self.inshape = inshape
        
        kernel_init = tf.keras.initializers.LecunNormal() #recommended for SELUs
        conv_opt = dict( strides=1,
                         kernel_size=kernel_size,
                         padding='same', #'valid' means no padding
                         activation='selu',
                         kernel_initializer=kernel_init,
                         use_bias=True )
        self.conv1 = Conv1D( input_shape=(self.inshape[0],1),
                             filters=16,
                             **conv_opt )
    
        self.dense1 = Dense( units=200,
                             activation='relu',
                             name='first dense')

        self.dense2 = Dense( units=self.inshape[0],
                             activation='selu',
                             name='second dense')

    #@debug_tensor_shape(name='x', run=False)
    def __call__(self, x):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.reshape(x, shape=(-1, x.shape[0]))
        x = self.dense1(x)
        x = tf.reshape(x, shape=(-1, x.shape[1], 1))
        x = self.conv1(x)
        x = tf.reshape(x, shape=(-1, x.shape[1]*x.shape[2]))
        x = self.dense2(x)
        x = tf.squeeze(x)
        return x

class TriggerCellDistributor(tf.Module):
    """Neural net workings"""
    def __init__( self, indata, inbins, bound_size,
                  kernel_size, window_size,
                  phibounds, nbinsphi, rzbounds, nbinsrz,
                  init_pars,
                  pretrained ):
        """
        Manages quantities related to the neural model being used.
        Args: 
              - kernel_size: Length of convolutional kernels
              - window_size: Number of bins considered for each variance calculation.
              Note this is not the same as number of trigger cells (each bin has
              multiple trigger cells).
        """        
        self.indata = indata
        self.boundary_size = bound_size
        self.kernel_size = kernel_size
        self.boundary_width = window_size-1
        self.phibounds, self.nbinsphi = phibounds, nbinsphi
        self.rzbounds, self.nbinsrz = rzbounds, nbinsrz

        self.pretrained = pretrained
        self.first_train = True

        self.architecture = Architecture(self.indata.shape, kernel_size)
        
        assert len(init_pars)==4
        self.init_pars = init_pars
        self.pars = init_pars

        self.subtract_max = lambda x: x - (tf.math.reduce_max(x)+1)
        # convert bin ids coming before 0 (shifted boundaries) to negative ones
        self.inbins = tensorflow_assignment( tensor=inbins,
                                             mask=tf.concat((tf.ones(self.boundary_size),
                                                             tf.zeros(inbins.shape[0]-self.boundary_size)), axis=0),
                                             lambda_op=self.subtract_max,
                                            )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-4)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.was_calc_loss_called = False

        # set checkpoint
        self.model_name = 'data/test_model/tf_checkpoints'
        self.checkpoint = tf.train.Checkpoint(model=self.architecture, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.model_name, max_to_keep=5)

    def adapt_loss_parameters(self, epoch):
        """Changes the proportionality between the loss terms as a function of the epoch."""
        return self.pars
        
    def calc_loss(self, originaldata, outdata):
        """
        Calculates the model's loss function. Receives slices in R/z as input.
        Each array value corresponds to a trigger cell.
        Args: -originaldata: similar to originaldata, but after postprocessing (denormalization).
                 To avoid confusions, `originaldata` should not be used in this method.
              -inbins: ordered (ascendent) original phi bins
              -bound_size: number of replicated trigger cells to satisfy boundary conditions
              -outdata: neural net postprocessed phi values output
        """
        opt = {'summarize': 10} #number of entries to print if assertion fails
        asserts = [ tf.debugging.Assert(originaldata.shape == outdata.shape, [originaldata.shape, outdata.shape], **opt) ]
        asserts.append( tf.debugging.Assert(tf.math.reduce_min(self.inbins)==-self.boundary_width, [self.inbins], **opt) )
        asserts.append( tf.debugging.Assert(tf.math.reduce_max(self.inbins)==self.nbinsphi-1, [self.inbins], **opt) )

        # bin the output of the neural network
        outbins = tf.histogram_fixed_width_bins(outdata,
                                                value_range=self.phibounds,
                                                nbins=self.nbinsphi)
        outbins = tf.cast(outbins, dtype=tf.float32)

        asserts.append( tf.debugging.Assert(tf.math.reduce_min(outbins)>=0, [outbins], **opt) )
        asserts.append( tf.debugging.Assert(tf.math.reduce_max(outbins)<=self.nbinsphi-1, [outbins], **opt) )

        # convert bin ids coming before 0 (shifted boundaries) to negative ones
        # if this looks convoluted, blame tensorflow, which still does not support tensor assignment
        # a new tensor must be created instead
        outbins = tensorflow_assignment( tensor=outbins,
                                         mask=tf.concat((tf.ones(self.boundary_size),
                                                         tf.zeros(outbins.shape[0]-self.boundary_size)), axis=0),
                                         lambda_op=self.subtract_max,
                                       )

        # replicated boundaries should be the same
        asserts.append( tf.debugging.Assert(
            condition=are_tensors_equal(originaldata[:self.boundary_size], originaldata[-self.boundary_size:]),
            data=[originaldata[:self.boundary_size],originaldata[-self.boundary_size:]],
            **opt) )
        
        with tf.control_dependencies(asserts):
            # print('Out: ', outdata.shape, outdata[:1])
            # print('In:  ', originaldata.shape, originaldata[:1])
            # print()

            equality_loss = tf.reduce_sum( tf.math.square(outdata-originaldata) )
            variance_loss = self._calc_local_variance(outdata, outbins)
            wasserstein_loss = tensorflow_wasserstein_1d_loss(originaldata, outdata)
            boundary_sanity_loss = tf.reduce_sum(
                tf.math.square( outdata[-self.boundary_size:] - outdata[:self.boundary_size] ) )

            loss_pars = [ tf.Variable(x, trainable=False) for x in self.pars ]

        return ( { 'equality_loss':       loss_pars[0] * equality_loss,
                   'local_variance_loss': loss_pars[1] * variance_loss,
                   'wasserstein_loss':    loss_pars[2] * wasserstein_loss,
                   'boundary_loss':       loss_pars[3] * boundary_sanity_loss },
                 tf.unique(outbins[:-self.boundary_size])[0]
                 )

    def _calc_local_variance(self, data, bins):
        """
        Calculates the variance between adjacent bins.
        Adjacent here refers to the bin index, and not necessarily to physical location
        (I had to take into account circular boundary conditions).
        """
        variance_loss = 0
        unique_bins = tf.unique(bins[:-self.boundary_size])[0]
        for ibin in unique_bins:
            idxs = (bins >= ibin) & (bins <= ibin+self.boundary_width)
            variance_loss += tf.math.reduce_variance(data[idxs])
        return variance_loss

    # @tf.function(
    #     input_signature=(
    #         tf.TensorSpec(shape=[], dtype=tf.float32),
    #         tf.TensorSpec(shape=[], dtype=tf.int32))
    # )
    #@tf.function
    def train_step(self, dp, save=False):
        # Rseset the metrics at the start of the next epoch
        self.train_loss.reset_states()

        # for n in tf.range(steps): # any future loop within one epoch is done here
        with tf.GradientTape() as tape:
            tape.watch(self.indata)
            tape.watch(self.inbins)

            if self.pretrained and self.first_train:
                #self.load_model()
                self.restore_checkpoint()
                self.first_train = False
                
            if save and not self.pretrained:
                self.save_checkpoint()

            prediction = self.architecture(self.indata)
            prediction = dp.postprocess(prediction)

            original_data = dp.postprocess(self.indata)

            losses, _ = self.calc_loss(original_data, prediction)
            loss_sum = tf.reduce_sum(list(losses.values()))

        gradients = tape.gradient(loss_sum, self.architecture.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.train_loss(loss_sum)
        return losses, prediction, gradients, self.architecture.trainable_variables

    def save_architecture_diagram(self, name):
        """Plots the structure of the used architecture."""
        assert name.split('.')[1] == 'png'
        tf.keras.utils.plot_model(
            self.architecture,
            to_file=name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )

    # def save_model(self):
        #self.architecture.save_weights(self.model_name)
        #self.architecture.save(self.model_name)

    def save_checkpoint(self):
        self.manager.save()

    def restore_checkpoint(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if not self.manager.latest_checkpoint:
            raise ValueError('Issue with checkpoint manager.')

    def load_model(self):
        #self.architecture.load_weights(self.model_name)
        self.architecture = tf.keras.models.load_model(self.model_name)
        

def save_scalar_logs(writer, scalar_map, epoch):
    """
    Saves tensorflow info for Tensorboard visualization.
    `scalar_map` expects a dict of (scalar_name, scalar_value) pairs.
    """
    with writer.as_default():
        tot = 0
        for k,v in scalar_map.items():
            tf.summary.scalar(k, v, step=epoch)
            tot += v
        tf.summary.scalar('total', tot, step=epoch)

def save_gradient_logs(writer, gradients, train_variables, epoch):
    """
    Saves tensorflow info for Tensorboard visualization.
    `scalar_map` expects a dict of (scalar_name, scalar_value) pairs.
    """
    with writer.as_default():
        for weights, grads in zip(train_variables, gradients):
            tf.summary.histogram(
                weights.name.replace(':', '_')+'_grads', data=grads, step=epoch)

def optimization(algo, **kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    plotter = Plotter(**optimization_kwargs)

    assert len(store_in.keys()) == 1
    dp = DataProcessing(phi_bounds=(kw['MinPhi'],kw['MaxPhi']))
    _, train_data, boundary_sizes = dp.preprocess( data=store_in['data'],
                                                   nbins_phi=kw['NbinsPhi'],
                                                   nbins_rz=kw['NbinsRz'],
                                                   window_size=kw['WindowSize'] )

    chosen_layer = 0
    orig_data = dp.postprocess(train_data[chosen_layer][:,0])

    plotter.save_orig_data(orig_data, boundary_sizes[chosen_layer])
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'

    for i,rzslice in enumerate(train_data):
        if i!=chosen_layer:  #look at the first R/z slice only
            continue
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        tcd = TriggerCellDistributor(
            indata=tf.convert_to_tensor(rzslice[:,0], dtype=tf.float32),
            inbins=tf.convert_to_tensor(rzslice[:,1], dtype=tf.float32),
            bound_size=boundary_sizes[i],
            kernel_size=kw['KernelSize'],
            window_size=kw['WindowSize'],
            phibounds=(kw['MinPhi'],kw['MaxPhi']),
            nbinsphi=kw['NbinsPhi'],
            rzbounds=(kw['MinROverZ'],kw['MaxROverZ']),
            nbinsrz=kw['NbinsRz'],
            init_pars=(1., 0., 1., 1.),
            pretrained=kw['Pretrained'],
        )
        #tcd.save_architecture_diagram('model{}.png'.format(i))

        for epoch in tqdm(range(kw['Epochs'])):
            should_save = True if epoch%20==0 else False

            tcd.adapt_loss_parameters(epoch)
            dictloss, outdata, gradients, train_vars = tcd.train_step(dp, save=should_save)

            save_scalar_logs(
                writer=summary_writer,
                scalar_map=dictloss,
                epoch=epoch
            )
            save_gradient_logs(
                writer=summary_writer,
                gradients=gradients,
                train_variables=train_vars,
                epoch=epoch
            )

            plotter.save_gen_data(outdata.numpy(), boundary_sizes[i])

            if should_save:
                plotter.plot(minval=-1, maxval=52, density=False, show_html=False)
        #plotter.plot(density=False, show_html=True)

if __name__ == "__main__":
    from airflow.airflow_dag import optimization_kwargs
    for falgo in optimization_kwargs['FesAlgos']:
        optimization( falgo, **optimization_kwargs )

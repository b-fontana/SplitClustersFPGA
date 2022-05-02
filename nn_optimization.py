import h5py
from tqdm import tqdm
import datetime
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

from data_processing import DataProcessing
from plotter import Plotter
from debug_architecture import debug_tensor_shape

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
    def __init__(self, inshape, nbins, kernel_size):
        super().__init__()
        assert len(inshape)==1

        self.inshape = inshape
        self.nbins = nbins
        
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
                             activation='selu',
                             name='first dense')

        self.dense2 = Dense( units=self.inshape[0],
                             activation='selu',
                             name='output data')

        self.dense3 = Dense( units=self.nbins,
                             activation='selu',
                             name='output bins')

    #@debug_tensor_shape(name='x', run=True)
    def __call__(self, x):
        # tf.summary.trace_on()
        x = tf.cast(x, dtype=tf.float32)
        x = tf.reshape(x, shape=(-1, x.shape[0], 1))
        x = self.conv1(x)
        x = tf.reshape(x, shape=(-1, x.shape[1]*x.shape[2]))
        x = self.dense1(x)
        x = tf.reshape(x, shape=(-1, x.shape[1]))
        x = self.dense2(x)
        outdata = tf.squeeze(x)

        x = self.dense3(x)
        outbins = tf.squeeze(x)

        return outdata, outbins

class TriggerCellDistributor(tf.Module):
    """Neural net workings"""
    def __init__( self,
                  indata,
                  inbins,
                  bound_size,
                  kernel_size,
                  window_size,
                  mode,
                  phibounds,
                  nbinsphi,
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
        self.inbins = inbins
        self.boundary_size = bound_size
        self.kernel_size = kernel_size
        self.phibounds = phibounds
        self.nbinsphi = nbinsphi

        self.pretrained = pretrained
        self.first_train = True

        self.local_loss_mode = mode
        self.boundary_width = window_size - 1

        self.architecture = Architecture(self.indata.shape, self.nbinsphi, kernel_size)
        
        self.pars = [None, None, None]
        assert len(self.pars) == 3

        self.subtract_max = lambda x: x - (tf.math.reduce_max(x)+1)
        # convert bin ids coming before 0 (shifted boundaries) to negative ones
        # self.inbins = tensorflow_assignment( tensor=inbins,
        #                                      mask=tf.concat((tf.ones(self.boundary_size),
        #                                                      tf.zeros(inbins.shape[0]-self.boundary_size)), axis=0),
        #                                      lambda_op=self.subtract_max,
        #                                     )

        if not self.pretrained:
            self.learning_rates = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7,)
            self.lr_thresholds = (0,400,700,1000,1300)
        else:
            self.learning_rates = (1e-7, 1e-6, 1e-5)
            self.lr_thresholds = (0, 100, 200)
        assert len(self.lr_thresholds)==len(self.learning_rates)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates[0])
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.was_calc_loss_called = False

        # set checkpoint
        self.model_name = os.path.join('data',
                                       'model_bound' + str(self.boundary_width),
                                       'tf_checkpoints')
        self.checkpoint = tf.train.Checkpoint(model=self.architecture, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.model_name, max_to_keep=5)

    def adapt_loss_parameters(self, epoch):
        """Changes the proportionality between the loss terms as a function of the epoch."""
        if self.pretrained: #1e-1, 1, 1 converges!
            initial_variance_loss_constant = 1.e-1
            initial_equal_data_loss_constant = 1.
            initial_equal_bins_loss_constant = 1.
        else:
            initial_variance_loss_constant = 0.
            initial_equal_data_loss_constant = 1.
            initial_equal_bins_loss_constant = 1.
            
        #variance_loss_constant = (1e-2, 1e-1, 1e-0, 1e1)
        variance_loss_constant = (initial_variance_loss_constant,)
        equal_data_loss_constant = tuple(initial_equal_data_loss_constant for _ in range(len(variance_loss_constant)))
        equal_bins_loss_constant = tuple(initial_equal_bins_loss_constant for _ in range(len(variance_loss_constant)))
        
        if self.pretrained:
            #thresholds = (100, 200, 300, 400,)
            loss_thresholds = (0,)
        else:
            loss_thresholds = (0,)

        assert len(loss_thresholds)==len(variance_loss_constant)
        
        # custom learning rate schedule
        for i in range(len(loss_thresholds)):

            #change the learning rate only at the threshold
            #let Adam schedule it during the rest of the time
            if epoch != loss_thresholds[i]:
                continue

            self.pars[0] = equal_data_loss_constant[i]
            self.pars[1] = equal_bins_loss_constant[i]
            self.pars[2] = variance_loss_constant[i]

        loss_pars = {'equal_data_loss_constant': self.pars[0],
                     'equal_bins_loss_constant': self.pars[1],
                     'variance_loss_constant': self.pars[2] }
        return loss_pars, self.pars

    def adapt_learning_rate(self, epoch):
        """
        Using the Adam optimizer the following will control the base learning rate, 
        not the effective one, which is adaptive.
        """
        learning_rates = (1e-4, 5e-5, 1e-5, 1e-6, 1e-7,)
        
        if not self.pretrained:
            thresholds = (300, 500, 700, 1000, 1300,)
            assert len(thresholds)==len(learning_rates)

            # make sure the initial learning rate is the largest
            for elem in learning_rates:
                assert self.init_lr > elem

            #change the learning rate only at the threshold
            #let Adam schedule it during the rest of the time
            if epoch != self.lr_thresholds[i]:
                continue

            self.optimizer.learning_rate.assign(self.learning_rates[i])

        return {'learning_rate': self.optimizer.lr}

    def calc_loss(self, indata, inbins, outdata, outbins):
        """
        Calculates the model's loss function. Receives slices in R/z as input.
        Each array value corresponds to a trigger cell.
        """
        inbins = tf.cast(inbins, dtype=tf.float32)
        outbins = tf.cast(outbins, dtype=tf.float32)

        opt = {'summarize': 10} #number of entries to print if assertion fails
        asserts = [ tf.debugging.Assert(indata.shape == outdata.shape, [indata.shape, outdata.shape], **opt) ]
        # asserts.append( tf.debugging.Assert(tf.math.reduce_min(inbins)==-self.boundary_width, [inbins], **opt) )
        # asserts.append( tf.debugging.Assert(tf.math.reduce_max(inbins)==self.nbinsphi-1, [inbins], **opt) )
                
        with tf.control_dependencies(asserts):
            # print(outdata)
            # print(indata)
            # print(outbins)
            # print(inbins)
            # print()
            # print(outdata.shape)
            # print(indata.shape)
            # print(outbins.shape)
            # print(inbins.shape)
            # quit()
            data_equality_loss = tf.reduce_sum( tf.math.square(outdata-indata) )
            bins_equality_loss = tf.reduce_sum( tf.math.square(outbins-inbins) )
            bins_variance_loss = self._calc_local_loss(outbins)

            loss_pars = []
            for par in self.pars:
                assert par is not None
                assert par >= 0.
                loss_pars.append( tf.Variable(par, trainable=False) )

        return { 'data_equality_loss':  loss_pars[0] * data_equality_loss,
                 'bins_equality_loss':  loss_pars[1] * bins_equality_loss,
                 'local_variance_loss': loss_pars[2] * bins_variance_loss }

    def _calc_local_loss(self, outbins):
        """
        Calculates the variance between adjacent bins.
        Adjacent here refers to the bin index, and not necessarily to physical location
        (I had to take into account circular boundary conditions).
        Bins without entries do not affect the calculation.
        """
        variance_loss = 0
        if self.local_loss_mode == 'variance':
            outbins_extended = tf.concat((outbins[-self.boundary_width:],
                                          outbins), axis=-1)
            frames = tf.signal.frame(outbins_extended,
                                     frame_length=self.boundary_width+1,
                                     frame_step=1)
            # the following is NOT the same as `tf.math.reduce_variance(frames)`
            var = tf.math.reduce_variance(frames, axis=-1)
            var = tf.reduce_sum(var)
            if not tf.math.is_nan(var):
                variance_loss = var
        elif self.local_loss_mode == 'diff':
            #take into account boundary conditions
            diff = tf.concat((tf.expand_dims(outbins[-1]-outbins[0], -1), outbins[1:]-outbins[:-1]), axis=0)
            diff = tf.reduce_sum(tf.math.square(diff))
            if not tf.math.is_nan(diff):
                variance_loss = diff
        else:
            m = 'Mode {} is not supported.'.format(mode)
            raise ValueError('[_calc_local_loss]' + m)
        return variance_loss

    def train_step(self, dp, save=False):
        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()

        # for n in tf.range(steps): # any future loop within one epoch is done here
        with tf.GradientTape() as tape:
            tape.watch(self.indata)
            #tape.watch(self.inbins)

            if self.pretrained and self.first_train:
                self.restore_checkpoint()
                self.first_train = False
                
            if save and not self.pretrained:
                self.save_checkpoint()

            prediction_data, prediction_bins = self.architecture(self.indata)
            losses = self.calc_loss(self.indata, self.inbins, prediction_data, prediction_bins)
            loss_sum = tf.reduce_sum(list(losses.values()))

        # print([ x.name for x in self.architecture.trainable_variables ], flush=True)
        gradients = tape.gradient(loss_sum, self.architecture.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.train_loss(loss_sum)

        prediction_data, prediction_bins = dp.postprocess( prediction_data,
                                                           prediction_bins )

        original_data, original_bins = dp.postprocess(self.indata, self.inbins)

        return losses, prediction_data, prediction_bins, gradients, self.architecture.trainable_variables

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
    are_losses = { 'data_equality_loss': 'equal_data_loss_constant',
                   'bins_equality_loss': 'equal_bins_loss_constant',
                   'local_variance_loss': 'variance_loss_constant' }
    
    with writer.as_default():
        tot = 0
        for k,v in scalar_map.items():
            if k in are_losses.keys():
                tot += v
                tf.summary.scalar(k, v / scalar_map[ are_losses[k] ], step=epoch)
            else:
                tf.summary.scalar(k, v, step=epoch)
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

# def save_graph(writer):
#     with writer.as_default():
#         tf.summary.trace_export('graph', 0)
        
def optimization(**kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    plotter = Plotter(**optimization_kwargs)
    mode = 'variance'
    window_size = 3 if mode == 'variance' else 2 if mode == 'diff' else 0

    assert len(store_in.keys()) == 1
    dp = DataProcessing( phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
                         bin_bounds=(0,50) )
    train_data, train_bins, _, _ = dp.preprocess( data=store_in['data'],
                                                  nbins_phi=kw['NbinsPhi'],
                                                  nbins_rz=kw['NbinsRz'],
                                                  window_size=window_size )
    chosen_layer = 0
    orig_data, orig_bins = dp.postprocess( train_data[chosen_layer][:,0],
                                           train_bins[chosen_layer] )

    plotter.save_orig_data(data=orig_data, data_type='data', boundary_sizes=0) #boundary_sizes[chosen_layer]
    plotter.save_orig_data(data=orig_bins, data_type='bins', boundary_sizes=0)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'

    for i,(rzdata,rzbins) in enumerate(zip(train_data, train_bins)):
        if i!=chosen_layer:  #look at the first R/z slice only
            continue
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        tcd = TriggerCellDistributor(
            indata=tf.convert_to_tensor(rzdata[:,0], dtype=tf.float32),
            inbins=tf.convert_to_tensor(rzbins, dtype=tf.float32),

            #bound_size=boundary_sizes[i],
            bound_size=0,
            kernel_size=kw['KernelSize'],
            window_size=window_size,
            mode=mode,
            phibounds = (kw['MinPhi'], kw['MaxPhi']),
            nbinsphi=kw['NbinsPhi'],
            pretrained=kw['Pretrained'],
        )
        #tcd.save_architecture_diagram('model{}.png'.format(i))

        for epoch in tqdm(range(kw['Epochs'])):
            should_save = True if epoch%20==0 else False

            loss_pars_map, _ = tcd.adapt_loss_parameters(epoch)
            lr_map = tcd.adapt_learning_rate(epoch)
            dictloss, outdata, outbins, gradients, train_vars = tcd.train_step(dp,
                                                                               save=should_save)
            
            scalar_map = dict(loss_pars_map)
            scalar_map.update(lr_map)
            scalar_map.update(dictloss)

            save_scalar_logs(
                writer=summary_writer,
                scalar_map=scalar_map,
                epoch=epoch
            )
            save_gradient_logs(
                writer=summary_writer,
                gradients=gradients,
                train_variables=train_vars,
                epoch=epoch
            )
            # save_graph(
            #     writer=summary_writer,
            # )

            plotter.save_gen_data(outdata.numpy(), boundary_sizes=0, data_type='data')
            plotter.save_gen_data(outbins.numpy(), boundary_sizes=0, data_type='bins')

            if should_save:
                plot_name = ( 'optimize_mode' + mode + '_ws' +
                              str(window_size) + '_pretrain' +
                              str(kw['Pretrained']) + '.html' )
                plotter.plot(plot_name=plot_name,
                             minval=-1, maxval=52,
                             density=False, show_html=False)

        #plotter.plot(density=False, show_html=True)

if __name__ == "__main__":
    #from airflow.airflow_dag import optimization_kwargs
    import os
    import numpy as np
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    Nevents = 16#{{ dag_run.conf.nevents }}
    NbinsRz = 42
    NbinsPhi = 216
    MinROverZ = 0.076
    MaxROverZ = 0.58
    MinPhi = -np.pi
    MaxPhi = +np.pi
    DataFolder = 'data'
    optimization_kwargs = { 'NbinsRz': NbinsRz,
                            'NbinsPhi': NbinsPhi,
                            'MinROverZ': MinROverZ,
                            'MaxROverZ': MaxROverZ,
                            'MinPhi': MinPhi,
                            'MaxPhi': MaxPhi,

                            'LayerEdges': [0,28],
                            'IsHCAL': False,

                            'Debug': True,
                            'DataFolder': DataFolder,
                            'FesAlgos': ['ThresholdDummyHistomaxnoareath20'],
                            'BasePath': os.path.join(os.environ['PWD'], DataFolder),

                            'RzBinEdges': np.linspace( MinROverZ, MaxROverZ, num=NbinsRz+1 ),
                            'PhiBinEdges': np.linspace( MinPhi, MaxPhi, num=NbinsPhi+1 ),
                            'Epochs': 99999,
                            'KernelSize': 10,
                            'OptimizationIn': os.path.join(os.environ['PWD'], DataFolder, 'triggergeom_condensed.hdf5'),
                            'OptimizationOut': 'None.hdf5',
                            'Pretrained': True,
                           }

    optimization( **optimization_kwargs )

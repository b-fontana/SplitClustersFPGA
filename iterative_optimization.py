import os
import random
import numpy as np
import h5py

from data_processing import DataProcessing
from plotter import Plotter
                
def optimization(**kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    plotter = Plotter(**optimization_kwargs)
    mode = 'variance'
    window_size = 3 # no boundaries added

    assert len(store_in.keys()) == 1
    dp = DataProcessing( phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
                         bin_bounds=(0,50) )
    data, bins, _, _ = dp.preprocess( data=store_in['data'],
                                      nbins_phi=kw['NbinsPhi'],
                                      nbins_rz=kw['NbinsRz'],
                                      window_size=window_size,
                                      normalize=False )
    chosen_layer = 0
  
    plotter.save_orig_data( data=np.array(data[chosen_layer][:,0]),
                            data_type='data',
                            boundary_sizes=0 )
    plotter.save_orig_data( data=np.array(bins),
                            data_type='bins',
                            boundary_sizes=0 )
    
    for i,(ldata,lbins) in enumerate(zip(data, bins)):
        if i!=chosen_layer:  #look at the first R/z slice only
            continue

        boundshift = window_size - 1
        ld = np.array(ldata)
        lb = np.array(lbins)

        idxs = [ np.arange(kw['NbinsPhi']) ]
        for _ in range(boundshift):
            idxs.append( np.roll(idxs[-1], -1) )

        # This algorithm is disgusting from a performance point of view!
        while True:
            for id_tuple in zip(*idxs):
                # bin indices
                id1, id2, id3 = id_tuple

                # bin counts
                c1, c2, c3 = lb[id1], lb[id2], lb[id3]

                # left and right gradients
                gl = c2 - c1
                gr = c3 - c2

                # weights for random draw
                gsum = abs(gl) + abs(gr)
                wl = abs(gl) / gsum
                assert( 0. <= wl <= 1.)
                wr = abs(gr) / gsum
                assert( 0. <= wr <= 1.)

                # "region" based on left and right gradients
                if gl <= 0 and gr >= 0:
                    region = 'valley'
                elif gl >= 0 and gr <= 0:
                    region = 'mountain'
                elif gl >= 0 and gr >= 0:
                    region = 'ascent'
                elif gl <= 0 and gr <= 0:
                    region = 'descent'
                else:
                    raise RuntimeError('Impossible 1!')
                
                # random draw (pick a side)
                side = 'left' if random.random() < wl else 'right'

                print(ld[id1,1])
                quit()
                if side == 'left' and region in ('valley', 'descent'):
                    lb[id1] -= 1
                    lb[id2] += 1
                elif side == 'right' and region in ('valley', 'ascent'):
                    lb[id3] -= 1
                    lb[id2] += 1
                elif side == 'left' and region in ('mountain', 'ascent'):
                    lb[id1] += 1
                    lb[id2] -= 1
                elif side == 'right' and region in ('mountain', 'descent'):
                    lb[id3] += 1
                    lb[id2] -= 1
                else:
                    raise RuntimeError('Impossible 2!')                    
                
            quit()

            
        # plotter.save_gen_data(outdata.numpy(), boundary_sizes=0, data_type='data')
        # plotter.save_gen_data(outbins.numpy(), boundary_sizes=0, data_type='bins')

        # plotter.plot(plot_name=plot_name,
        #              minval=-1, maxval=52,
        #              density=False, show_html=False)

if __name__ == "__main__":  
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
                            'OptimizationIn': os.path.join(os.environ['PWD'], DataFolder, 'triggergeom_condensed.hdf5'),
                            'OptimizationOut': 'None.hdf5',
                           }

    optimization( **optimization_kwargs )

"""
Some plots.
"""
import os
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, show, save

class Plotter:
    def __init__(self, outname='plots/plotter_out.hdf5', **kw):
        self.outname = outname
        self.kw = kw
        self.histos = []
        self.gen_data = []
        self.gen_data_counts = []
        self.gen_bins_counts = []
        self.orig_data = None
        self.orig_data_counts = None
        self.orig_bins = None

        self.remove_output()

        self.plot_min = 1e10
        self.plot_max = 0
        
    def remove_output(self):
        if os.path.exists(self.outname):
            os.system('rm {}'.format(self.outname))
        self.gen_data_counts = []
        self.histos = []

    def save_orig_data(self, data, data_type, boundary_sizes, bins=None, minlength=None):
        """
        Plots the original number of trigger cells per bin.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        assert data_type=='data' or data_type=='bins'
        
        if isinstance(data, (tuple,list)):
            raise ValueError('This method cannot be applied to a data list.')

        data = self._remove_duplicated_boundaries(data, boundary_sizes)

        if data_type == 'data':
            self.orig_data = data
            bins = self.kw['PhiBinEdges'] if bins is None else bins
            minlength = self.kw['NbinsPhi'] if minlength is None else minlength

            digi_bins = np.digitize(data, bins=bins)
            bincounts = np.bincount(digi_bins, minlength=minlength)
            assert bincounts[0] == 0 #no out-of-bound values are possible
            bincounts = bincounts[1:]

            assert len(bincounts) == self.kw['NbinsPhi']

            self.orig_data_counts = bincounts

        elif data_type == 'bins':
            self.orig_bins = data

    def _remove_duplicated_boundaries(self, data, bound_size):
        return data[bound_size:]
        
    def save_gen_data(self, data, boundary_sizes, data_type, bins=None, minlength=None):
        """
        Plots the number of trigger cells per bin as output by the learning framework.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        assert data_type=='data' or data_type=='bins'
        
        if isinstance(data, (tuple,list)):
            raise ValueError('Ups.')

        data = self._remove_duplicated_boundaries(data, boundary_sizes)

        if data_type == 'data':
            self.gen_data.append(data)
        elif data_type == 'bins':
            self.gen_bins_counts.append(data)

        if data_type == 'data':
            #find the overall min and max of the histogram to be plotted
            bins = self.kw['PhiBinEdges'] if bins is None else bins

            #consider the two out-of-bound bins
            minlength = self.kw['NbinsPhi']+2 if minlength is None else minlength

            _digi_bins = np.digitize(data, bins=bins)        
            _bincounts = np.bincount(_digi_bins, minlength=minlength)
            _bincounts = _bincounts[1:-1] #do not plot out-of-bound values
        elif data_type == 'bins':
            _bincounts = data
            
        if self.plot_max < np.max(_bincounts):
            self.plot_max = np.max(_bincounts)
        if self.plot_min > np.min(_bincounts):
            self.plot_min = np.min(_bincounts)

        if data_type == 'data':
            digi_bins = np.digitize(data, bins=bins)
            bincounts = np.bincount(digi_bins, minlength=minlength)
            assert len(bincounts)==minlength
            assert sum(self.orig_data_counts) >= sum(bincounts)

            #remove the out-of-bounds from the plot
            bincounts = bincounts[1:-1]
            self.gen_data_counts.append( bincounts )
            self.histos.append( np.histogram(bincounts, bins=20) )

    def plot(self, plot_name='plots/ntriggercells.html',
             minval=None, maxval=None,
             density=False, show_html=False):
        """Plots the data collected by the save_* methods."""
        assert self.orig_data is not None
        assert self.orig_bins is not None
        assert self.orig_data_counts is not None
        assert len(self.gen_data_counts) > 0
        assert len(self.gen_bins_counts) > 0
        logpad = 1e-4

        # if the out bin number is larger than `NbinsPhi` (makes no sense but can happen)
        # the shapes will not match
        nelems_max = 0
        for bc in self.gen_data_counts:
            if len(bc) > nelems_max:
                nelems_max = len(bc)

        binspad = np.pad(self.gen_data_counts[0],
                         pad_width=(0,nelems_max-len(self.gen_data_counts[0])))

        # data counts start display
        s_data_counts = { 'curr_x': np.arange(nelems_max) }
        if density:
            s_data_counts.update( {'curr_y': binspad/sum(binspad) + logpad} )
        else:
            s_data_counts.update( {'curr_y': binspad} )

        # bin counts start display
        gen0 = self.gen_bins_counts[0]
        s_bins_counts = { 'curr_x': np.arange(len(gen0)) }
        if density:
            s_bins_counts.update( {'curr_y': gen0/sum(gen0) + logpad} )
        else:
            s_bins_counts.update( {'curr_y': gen0} )

        nbins_plot = 150
        s_diff = {}
        hist, edges = np.histogram(self.orig_data-self.gen_data[0],
                                   density=False, bins=nbins_plot)
        df_diff = pd.DataFrame( {'diff_default': hist,
                                 'leftedge_default': edges[1:],
                                 'rightedge_default': edges[:-1]} )

        hists, lefts, rights = ([] for _ in range(3))
        for i,elem in enumerate(self.gen_data):
            hist, edges = np.histogram(self.orig_data-elem,
                                       density=False, bins=nbins_plot)
            hists.append( pd.DataFrame(hist, columns=['diff'+str(i)]) )
            lefts.append( pd.DataFrame(edges[1:], columns=['leftedge'+str(i)]) )
            rights.append( pd.DataFrame(edges[:-1], columns=['rightedge'+str(i)]) )

        hists_to_concat = [df_diff]
        hists_to_concat.extend(hists)
        hists_to_concat.extend(lefts)
        hists_to_concat.extend(rights)
        df_diff = pd.concat(hists_to_concat, axis=1)
        s_diff = ColumnDataSource(df_diff)
        
        for i,bc in enumerate(self.gen_data_counts):
            bc_padded = np.pad(bc, pad_width=(0,nelems_max-len(bc)))
            if density:
                s_data_counts.update({'bc'+str(i): (bc_padded)/sum(bc_padded) + logpad})
            else:
                s_data_counts.update({'bc'+str(i): (bc_padded)})

        for i,bc in enumerate(self.gen_bins_counts):
            if density:
                s_bins_counts.update({'bc'+str(i): (bc)/sum(bc) + logpad})
            else:
                s_bins_counts.update({'bc'+str(i): (bc)})

        s_data_counts = ColumnDataSource(data=s_data_counts)
        s_bins_counts = ColumnDataSource(data=s_bins_counts)
        
        y_axis_type = 'log' if density else 'linear'
        plot_opt = dict(width=1200, height=300)
        if minval is None or maxval is None:
            if density: 
                plot_distr = figure( y_range=(logpad,1), y_axis_type=y_axis_type,
                                     **plot_opt)
            else:
                plot_distr = figure( y_axis_type=y_axis_type, **plot_opt)
                #(self.plot_min,self.plot_max)
        else:
            y_range = (logpad,1) if density else (minval,maxval)
            plot_distr = figure( y_range=y_range, y_axis_type=y_axis_type,
                               **plot_opt)

        plot_distr.circle('curr_x', 'curr_y', source=s_data_counts,
                          legend_label='Data Output', color='red')
        plot_distr.triangle('curr_x', 'curr_y', source=s_bins_counts,
                          legend_label='Bins Output', color='orange')
        plot_distr.legend.click_policy='hide'

        callback = CustomJS(args=dict(s1=s_data_counts, s2=s_diff, s3=s_bins_counts),
                            code="""
                            const d1 = s1.data;
                            const d2 = s2.data;
                            const d3 = s3.data;
                            const slval = cb_obj.value;
                            const numberstr = slval.toString();
                            d1['curr_y'] = d1['bc'+numberstr];
                            d3['curr_y'] = d3['bc'+numberstr];
                            d2['diff_default'] = d2['diff'+numberstr];
                            d2['leftedge_default'] = d2['leftedge'+numberstr];
                            d2['rightedge_default'] = d2['rightedge'+numberstr];
                            s1.change.emit();
                            s2.change.emit();
                            s3.change.emit();
                            """)

        # some generated data can lie outside the bin leftmost and rightmost edges
        assert sum(self.orig_data_counts) >= sum(self.gen_data_counts[0]) 

        plot_distr.circle(np.arange(len(self.orig_data_counts)),
                          #self.orig_data_counts/sum(self.orig_data_counts),
                          self.orig_data_counts,
                          color='black',
                          legend_label='Data Input')

        plot_distr.triangle(np.arange(len(self.orig_bins)),
                            self.orig_bins,
                            color='grey',
                            legend_label='Bins Input')

        plot_diff = figure(**plot_opt)
        plot_diff.quad( top='diff_default', bottom=0,
                        left='leftedge_default', right='rightedge_default',
                        fill_color='navy', line_color='white', alpha=0.5,
                        source=s_diff )

        slider = Slider(start=0, end=len(self.gen_data),
                        value=0, step=1., title='Epoch')
        slider.js_on_change('value', callback)

        layout = column( slider, plot_distr, plot_diff )
        if show_html:
            show(layout)
        else:
            save(layout)
            

if __name__ == "__main__":
    import h5py
    from airflow.airflow_dag import optimization_kwargs
    from data_processing import preprocess
    
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')

    assert len(store_in.keys()) == 1
    data, _, _ = preprocess(
        data=store_in['data'],
        nbins_phi=kw['NbinsPhi'],
        nbins_rz=kw['NbinsRz'],
        window_size=kw['WindowSize']
    )
    rzslices = [ rzslice[:,0] for rzslice in data ]
    assert len(optimization_kwargs['FesAlgos'])==1
    plotter = Plotter()
    plotter.save( rzslices, save=False, **optimization_kwargs )
    plotter.plot()

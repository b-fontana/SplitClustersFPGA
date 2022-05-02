"""
Functions used for data processing.
"""
import numpy as np

class DataProcessing:
    def __init__(self, phi_bounds, bin_bounds):
        self.max_bound = 1.
        self.min_bound = 0.
        self.diff_bound = self.max_bound-self.min_bound
        self.shift_phi = np.pi #shift the phi range to [0; 2*Pi[
        self.phi_bounds = tuple( x + self.shift_phi for x in phi_bounds )

        #get parameters for the linear transformation (set between 0 and 1)
        self.a_norm_phi = self.diff_bound / (self.phi_bounds[1]-self.phi_bounds[0])
        self.b_norm_phi = self.max_bound - self.a_norm_phi * self.phi_bounds[1]

        self.bin_bounds = bin_bounds
        self.a_norm_bin = self.diff_bound / (self.bin_bounds[1] - self.bin_bounds[0])
        self.b_norm_bin = self.max_bound - self.a_norm_bin * self.bin_bounds[1]
        
    def preprocess( self, data,
                    nbins_phi,
                    nbins_rz,
                    window_size,
                    normalize=True ):
        """Prepare the data to serve as input to the net in R/z slices"""
        # data variables' indexes
        assert data.attrs['columns'].tolist() == ['Rz', 'phi', 'Rz_bin', 'phi_bin']

        rz_idx = 0
        phi_idx = 1
        rzbin_idx = 2
        phibin_idx = 3

        data = data[()] #eager (lazy is the default)

        def _drop_columns_data(data, data_with_boundaries, idxs):
            """Drops the columns specified by indexes `idxs`, overriding data arrays."""
            drop = lambda d,obj: np.delete(d, obj=obj, axis=1)
            data = [drop(x, idxs) for x in data]
            data_with_boundaries = [drop(x, idxs) for x in data_with_boundaries]
            return data, data_with_boundaries

        def _shift_data(data, index):
            """
            Shift data. Originally meant to avoid negative values.
            """
            if index == phi_idx:
                data[:,index] += self.shift_phi
            assert len(data[:,index][ data[:,index] < self.min_bound ]) == 0
            return data

        def _split_data(data, split_index, sort_index, nbins_rz):
            """
            Creates a list of R/z slices, each ordered by phi.
            """
            data = data.astype('float32')

            # data sanity check
            rz_slices = np.unique(data[:,split_index])

            assert len(rz_slices) <= nbins_rz
            assert rz_slices.tolist() == [x for x in range(len(rz_slices))]

            # Https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
            # ordering is already done when the data is produced, the following line is not needed anymore
            # data = data[ data[:,split_index].argsort(kind='stable') ] # sort rows by Rz_bin "column"

            # https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array

            # `np.diff` catches all `data` indexes where the sorted bin changes
            data = np.split( data, np.where(np.diff(data[:,sort_index])<0)[0]+1 )
            assert len(data) == len(rz_slices)

            # data correct sorting check
            for elem in data:
                assert ( np.sort(elem[:,sort_index], kind='stable') == elem[:,sort_index] ).all()

            return data

        def _normalize_data(data, index):
            """
            Standard max-min normalization of column `index`.
            """
            if index == phi_idx:
                data[:,index] = self.a_norm_phi * data[:,index] + self.b_norm_phi
            else:
                raise ValueError()
            return data

        def _set_boundary_conditions_data(data, window_size, phibin_idx, nbins_phi):
            """
            Pad the original data to ensure boundary conditions over its Phi dimension.
            The boundary is done in terms of bins, not single trigger cells.
            `bound_cond_width` stands for the number of bins seen to the right and left.
            The right boundary is concatenated on the left side of `data`.
            """
            bound_cond_width = window_size - 1
            boundary_right_indexes = [ (x[:,phibin_idx] >= nbins_phi-bound_cond_width)
                                       for x in data ]
            boundary_right = [ x[y] for x,y in zip(data,boundary_right_indexes) ]
            boundary_sizes = [ len(x) for x in boundary_right ]
            data_with_boundaries = [ np.concatenate((br,x), axis=0)
                                     for br,x in zip(boundary_right,data) ]

            return data, data_with_boundaries, boundary_sizes

        if normalize:
            data = _shift_data(
                data,
                index=phi_idx
            )
            data = _normalize_data(
                data,
                index=phi_idx
            )
        data = _split_data(
            data,
            split_index=rzbin_idx,
            sort_index=phibin_idx,
            nbins_rz=nbins_rz
        )
        data, data_with_boundaries, boundary_sizes = _set_boundary_conditions_data(
            data,
            window_size,
            phibin_idx,
            nbins_phi
        )
        data, data_with_boundaries = _drop_columns_data(
            data,
            data_with_boundaries,
            idxs=[rz_idx, rzbin_idx]
        )

        bins = []
        for rzslice in data:
            tmp = rzslice[:,1].astype(int)
            tmp = np.bincount( tmp )

            #normalization
            if normalize:
                tmp = self.a_norm_bin * tmp + self.b_norm_bin
            
            bins.append(tmp)

        return data, bins, data_with_boundaries, boundary_sizes

    def postprocess(self, data, bins):
        """Adapt the output of the neural network."""

        def _denormalize(data, bins):
            """Opposite of preprocess._normalize()."""
            new_data = (data - self.b_norm_phi) / self.a_norm_phi
            new_bins = (bins - self.b_norm_bin) / self.a_norm_bin
            return new_data, new_bins

        def _deshift(data, bins):
            """Opposite of preprocess._normalize()."""
            new_data = data - self.shift_phi
            new_bins = bins
            return new_data, new_bins

        new_data, new_bins = _denormalize(
            data,
            bins,
        )
        new_data, new_bins = _deshift(
            new_data,
            new_bins,
        )

        return new_data, new_bins

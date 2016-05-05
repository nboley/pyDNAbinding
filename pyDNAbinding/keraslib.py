from collections import defaultdict

import numpy as np

import keras

import keras.backend as K
import keras.callbacks
from keras import initializations

from keras.models import Graph, Sequential, model_from_yaml
from keras.optimizers import Adam, Adamax, RMSprop, SGD

from keras.layers.core import (
    Activation, Dense, Dropout, Flatten, 
    Lambda, Layer, Merge, MaxoutDense, Permute, Reshape
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import theano
import theano.tensor as TT

from binding_model import ConvolutionalDNABindingModel
from misc import calc_occ, R, T

def cast_if_1D(x):
    if len(x.shape) == 1:
        return x[:,None]
    return x

def theano_calc_log_occs(affinities, chem_pot):
    inner = (-chem_pot+affinities)/(R*T)
    lower = TT.switch(inner<-10, TT.exp(inner), 0)
    mid = TT.switch((inner >= -10)&(inner <= 35), 
                    TT.log(1.0 + TT.exp(inner)),
                    0 )
    upper = TT.switch(inner>35, inner, 0)
    return -(lower + mid + upper)

def theano_logistic(affinities, chem_pot):
    return 1/(1+TT.exp(affinities + chem_pot))

def theano_log_sum_log_occs(log_occs):
    # theano.printing.Print('scale_factor')
    scale_factor = (
        TT.max(log_occs, axis=1, keepdims=True))
    centered_log_occs = (log_occs - scale_factor)
    centered_rv = TT.log(TT.sum(TT.exp(centered_log_occs), axis=1))
    return centered_rv + scale_factor.flatten()

def iter_weighted_batch_samples(model, batch_iterator, oversampling_ratio=1):
    """Iter batches where poorly predicted samples are more frequently selected.

    model: the model to predict from
    batch_iterator: batch iterator to draw samples from
    oversampling_ratio: how many batch_iterator batches to sample from
    """
    assert oversampling_ratio > 0

    while True:
        # group the output of oversampling_ratio batches together
        all_batches = defaultdict(list)
        for i in xrange(oversampling_ratio):
            for key, val in next(batch_iterator).iteritems():
                all_batches[key].append(val)
        # find the batch size
        batch_size = next(all_batches.itervalues())[0].shape[0]
        # stack the output
        for key, batches in all_batches.iteritems():
            all_batches[key] = np.vstack(batches)

        # weight the batch values
        pred_vals = model.predict_on_batch(all_batches)
        weights = np.zeros(batch_size*oversampling_ratio)
        for key in pred_vals:
            if key.endswith('binding_occupancies'): continue
            losses = (pred_vals[key] - all_batches[key])**2
            # set ambiguous labels to 0
            losses[all_batches[key] < -0.5] = 0
            # add to the losses,
            inner_weights = losses.sum(1)
            # re-weight the rows with ambiguous labels
            inner_weights *= losses.shape[1]/((
                all_batches[key] > -0.5).sum(1) + 1e-6)
            # make sure that every row has some weight
            inner_weights += 1e-6
            weights += inner_weights
        # normalize the weights to 1
        weights /= weights.sum()

        # downsample batch_size observations
        output = {}
        loss_indices = np.random.choice(
            len(weights), size=batch_size, replace=False, p=weights)
        for key in all_batches.keys():
            output[key] = all_batches[key][loss_indices,:]
        yield output
    
    return

class ConvolutionDNAShapeBinding(Layer):
    def __init__(
            self, nb_motifs, motif_len, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.input = K.placeholder(ndim=4)
        
        self.init = lambda x: (
            initializations.get(init)(x), 
            K.zeros((self.nb_motifs,)) 
        )
        super(ConvolutionDNAShapeBinding, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_motifs': self.nb_motifs,
                  'motif_len': self.motif_len, 
                  'init': self.init.__name__}
        base_config = super(ConvolutionDNAShapeBinding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self):
        input_dim = self.input_shape[1]
        self.W_shape = (
            self.nb_motifs, 1, self.motif_len, 6)
        self.W, self.b = self.init(self.W_shape)
        self.trainable_weights = [self.W, self.b]
    
    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0], 
                self.nb_motifs,
                # sequence length minus the motif length
                self.input_shape[2], #-self.motif_len+1,
                2)
    
    def get_output(self, train=False):
        X = self.get_input(train)
        fwd_rv = K.conv2d(X[:,:,:,:6], 
                          self.W, 
                          border_mode='valid') \
                 + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X[:,:,:,-6:][:,::-1,::-1], 
                         self.W, 
                         border_mode='valid') \
                + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        return K.concatenate((fwd_rv, rc_rv), axis=3)
        #return rc_rv

class ConvolutionDNASequenceBinding(Layer):
    def __init__(
            self,
            nb_motifs,
            motif_len, 
            use_three_base_encoding=True,
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.input = K.placeholder(ndim=4)
        self.use_three_base_encoding = use_three_base_encoding
        self.kwargs = kwargs
        
        self.W = None
        self.b = None
        
        #if isinstance(init, ConvolutionalDNABindingModel):
        #    self.init = lambda x: (
        #        K.variable(-init.ddg_array[None,None,:,:]), 
        #        K.variable(np.array([-init.ref_energy,])[:,None]) 
        #    )
        #else:
        #    self.init = lambda x: (
        #        initializations.get(init)(x), 
        #        K.zeros((self.nb_motifs,)) 
        #    )
        self.init = initializations.get(init)
        super(ConvolutionDNASequenceBinding, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_motifs': self.nb_motifs,
                  'motif_len': self.motif_len, 
                  'use_three_base_encoding': self.use_three_base_encoding,
                  'init': self.init.__name__}
        base_config = super(ConvolutionDNASequenceBinding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def create_clone(self):
        """
        
        """
        rv = type(self)(
            self.nb_motifs, 
            self.motif_len, 
            self.use_three_base_encoding, 
            self.init,
            **self.kwargs)
        rv.W = self.W
        rv.b = self.b
        rv.W_shape = self.W_shape
        return rv

    def init_filters(self):
        if self.use_three_base_encoding:
            self.W_shape = (
                self.nb_motifs, 3, 1, self.motif_len)
        else:
            self.W_shape = (
                self.nb_motifs, 4, 1, self.motif_len)
        self.W = self.init(self.W_shape) 
        self.b = K.zeros((self.nb_motifs,))
        return

    def build(self):
        #print "Small Domains W SHAPE:", self.W_shape
        assert self.input_shape[1]==4, "Expecting one-hot-encoded DNA sequence."
        if self.W is None:
            assert self.b is None
            self.init_filters()
        self.trainable_weights = [self.W[0], self.b[0]]

    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0],
                self.nb_motifs,
                2,
                # sequence length minus the motif length
                self.input_shape[3]-self.motif_len+1,
                )
    
    def get_output(self, train=False):
        print "Input Shape", self.input_shape
        print "ConvolutionDNASequenceBinding", self.output_shape
        X = self.get_input(train)
        if self.use_three_base_encoding:
            X_fwd = X[:,1:,:,:]
            X_rc = X[:,:3,:,:]
        else:
            X_fwd = X
            X_rc = X

        print self.W
        print self.b
        if self.W[1] is not None:
            W = self.W[0][self.W[1],:,:,:]
        else:
            W = self.W[0]
        if self.b[1] is not None:
            b = self.b[0][self.b[1]]
        else:
            b = self.b[0]
        
        fwd_rv = K.conv2d(X_fwd, W, border_mode='valid') \
                 + K.reshape(b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X_rc, W[:,::-1,:,::-1], border_mode='valid') \
                + K.reshape(b, (1, self.nb_motifs, 1, 1))
        rv = K.concatenate((fwd_rv, rc_rv), axis=2)            
        #return rv.dimshuffle((0,3,2,1))
        return rv # K.permute_dimensions(rv, (0,3,2,1))

    def extract_binding_models(self):
        mos = []
        for i in xrange(self.nb_motifs):
            ddg_array = self.W[0].get_value()[i,:,0,:]
            ddg_array = np.vstack(
                (np.zeros((1, ddg_array.shape[1])), ddg_array)).T
            ref_energy = self.b[0].get_value()[i]
            mos.append(EnergeticDNABindingModel(-ref_energy, -ddg_array))
        return mos


class ConvolutionBindingSubDomains(Layer):
    def __init__(
            self,
            nb_domains, 
            domain_len, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_domains = nb_domains
        self.domain_len = domain_len

        self.input = K.placeholder(ndim=4)
        
        self.init = lambda x: (
            (initializations.get(init)(x), None),
            (K.zeros((self.nb_domains,)), None) 
        )
        self.kwargs = kwargs
        
        self.W_shape = None
        self.W = None
        self.b = None
        
        super(ConvolutionBindingSubDomains, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_domains': self.nb_domains,
                  'domain_len': self.domain_len, 
                  'init': self.init.__name__}
        base_config = super(ConvolutionBindingSubDomains, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def create_clone(self):
        """
        
        """
        rv = type(self)(
            self.nb_domains, 
            self.domain_len, 
            self.init,
            **self.kwargs)
        rv.W = self.W
        rv.b = self.b
        rv.W_shape = self.W_shape
        return rv

    def init_filters(self, num_input_filters):
        self.W_shape = (
            self.nb_domains, num_input_filters, 1, self.domain_len)
        self.W, self.b = self.init(self.W_shape)
        return

    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        assert self.input_shape[2] == 2
        if self.W is None:
            assert self.b is None
            self.init_filters(self.input_shape[1])
        print "Subdomains Filter Shape:", self.W_shape
        #assert self.input_shape[3] == self.W_shape[3]
        self.trainable_weights = [self.W[0], self.b[0]]

    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0], 
                self.nb_domains,
                # sequence length minus the motif length
                2,
                self.input_shape[3]-self.domain_len+1)
    
    def get_output(self, train=False):
        print "ConvolutionBindingSubDomains", self.output_shape
        X = self.get_input(train)
        if self.W[1] is not None:
            W = self.W[0][self.W[1],:,:,:]
        else:
            W = self.W[0]
        if self.b[1] is not None:
            b = self.b[0][self.b[1]]
        else:
            b = self.b[0]
        fwd_rv = K.conv2d(X[:,:,0:1,:], W, border_mode='valid')  \
                 + K.reshape(b, (1, self.nb_domains, 1, 1))
        # # [:,:,::-1,::-1]
        rc_rv = K.conv2d(X[:,:,1:2,:], W[:,:,:,::-1], border_mode='valid') \
                + K.reshape(b, (1, self.nb_domains, 1, 1))
        rv = K.concatenate((fwd_rv, rc_rv), axis=2)
        #return rv.dimshuffle((0,3,2,1))
        return rv #K.permute_dimensions(rv, (0,3,2,1))

class LogNormalizedOccupancy(Layer):
    def __init__(
            self, 
            init_chem_affinity=0.0, 
            steric_hindrance_win_len=None, 
            **kwargs):
        self.input = K.placeholder(ndim=4)
        self.init_chem_affinity = init_chem_affinity
        self.steric_hindrance_win_len = (
            0 if steric_hindrance_win_len is None 
            else steric_hindrance_win_len
        )
        super(LogNormalizedOccupancy, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(LogNormalizedOccupancy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        #assert self.input_shape[3] == 2
        self.chem_affinity = K.variable(self.init_chem_affinity)
        self.trainable_weights = [self.chem_affinity]

    @property
    def output_shape(self):
        #return self.input_shape
        assert self.input_shape[2] == 2
        return (# number of obseravations
                self.input_shape[0], 
                2*self.input_shape[1],
                # sequence length minus the motif length
                1, #-self.domain_len+1,
                self.input_shape[3] #-2*(self.steric_hindrance_win_len-1)
        )
    
    def get_output(self, train=False):
        print "LogNormalizedOccupancy", self.output_shape
        X = self.get_input(train)
        # calculate the log occupancies
        log_occs = theano_calc_log_occs(-X, self.chem_affinity)
        # reshape the output so that the forward and reverse complement 
        # occupancies are viewed as different tracks 
        log_occs = K.reshape(log_occs, (X.shape[0], 1, 2*X.shape[1], X.shape[3]))
        if self.steric_hindrance_win_len == 0:
            log_norm_factor = 0
        else:
            # correct occupancies for overlapping binding sites
            occs = K.exp(log_occs)
            kernel = K.ones((1, 1, 1, 2*self.steric_hindrance_win_len-1), dtype='float32')
            win_occ_sum = K.conv2d(occs, kernel, border_mode='same').sum(axis=2, keepdims=True)
            win_prb_all_unbnd = TT.exp(
                K.conv2d(K.log(1-occs), kernel, border_mode='same')).sum(axis=2, keepdims=True)
            log_norm_factor = TT.log(win_occ_sum + win_prb_all_unbnd)
        #start = max(0, self.steric_hindrance_win_len-1)
        #stop = min(self.output_shape[3], 
        #           self.output_shape[3]-(self.steric_hindrance_win_len-1))
        #rv = log_occs[:,:,:,start:stop] - log_norm_factor
        rv = (log_occs - log_norm_factor)
        return K.reshape(
            rv, 
            (X.shape[0], 2*X.shape[1], 1, X.shape[3])
        )

class TrackMax(Layer):
    def __init__(self, **kwargs):
        self.input = K.placeholder(ndim=4)        
        super(TrackMax, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(TrackMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output_shape(self):
        #return self.input_shape
        return (self.input_shape[0],
                self.input_shape[1],
                self.input_shape[2],
                1)

    def get_output(self, train=False):
        print "TrackMax", self.output_shape
        X = self.get_input(train)
        rv = K.max(X, axis=3, keepdims=True)
        return rv

class OccMaxPool(Layer):
    def __init__(self, num_tracks, num_bases, **kwargs):
        self.num_tracks = num_tracks
        self.num_bases = num_bases
        self.input = K.placeholder(ndim=4)        
        super(OccMaxPool, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'name': self.__class__.__name__, 
            'num_tracks': self.num_tracks, 
            'num_bases': self.num_bases
        }
        base_config = super(OccMaxPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output_shape(self):
        #return self.input_shape
        assert self.input_shape[2] == 1
        num_output_tracks = (
            1 if self.num_tracks == 'full' 
            else self.input_shape[1]//self.num_tracks )
        num_output_bases = (
            1 if self.num_bases == 'full' 
            else self.input_shape[3]//self.num_bases )
        return (
            self.input_shape[0],
            num_output_tracks, # + (
            #    1 if self.input_shape[1]%self.num_tracks > 0 else 0),
            1,
            num_output_bases# + (
            #    1 if self.input_shape[3]%self.num_bases > 0 else 0)
        )

    def get_output(self, train=False):
        print "OccMaxPool", self.output_shape, self.input_shape
        num_tracks = (
            self.input_shape[1] if self.num_tracks == 'full' 
            else self.num_tracks
        )
        num_bases = (
            self.input_shape[3] if self.num_bases == 'full' 
            else self.num_bases
        )
        X = self.get_input(train)
        X = K.permute_dimensions(X, (0,2,1,3))
        rv = K.pool2d(
            X, 
            pool_size=(num_tracks, num_bases), 
            strides=(num_tracks, num_bases),
            pool_mode='max'
        )
        rv = K.permute_dimensions(rv, (0,2,1,3))
        return rv

class ConvolutionCoOccupancy(Layer):
    def __init__(
            self,
            nb_domains, 
            domain_len, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_domains = nb_domains
        self.domain_len = domain_len

        self.input = K.placeholder(ndim=4)
        
        self.init = lambda x: (
            (initializations.get(init)(x), None),
            (K.zeros((self.nb_domains,)), None) 
        )
        self.kwargs = kwargs
        
        self.W_shape = None
        self.W = None
        self.b = None
        
        super(ConvolutionCoOccupancy, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_domains': self.nb_domains,
                  'domain_len': self.domain_len, 
                  'init': self.init.__name__}
        base_config = super(ConvolutionCoOccupancy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def create_clone(self):
        """
        
        """
        rv = type(self)(
            self.nb_domains, 
            self.domain_len, 
            self.init,
            **self.kwargs)
        rv.W = self.W
        rv.b = self.b
        rv.W_shape = self.W_shape
        return rv

    def init_filters(self, num_input_filters):
        self.W_shape = (
            self.nb_domains, num_input_filters, 1, self.domain_len)
        self.W, self.b = self.init(self.W_shape)
        return

    def build(self):
        # make sure that every occupancy has been flattenned into a single track
        assert self.input_shape[2] == 1
        if self.W is None:
            assert self.b is None
            self.init_filters(self.input_shape[1])
        print "Occ Subdomains Filter Shape:", self.W_shape
        #assert self.input_shape[3] == self.W_shape[3]
        self.trainable_weights = [self.W[0], self.b[0]]

    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0], 
                self.nb_domains,
                # sequence length minus the motif length
                1,
                self.input_shape[3]-self.domain_len+1)
    
    def get_output(self, train=False):
        print "ConvolutionCoOccupancy", self.output_shape, self.input_shape
        X = self.get_input(train)
        if self.W[1] is not None:
            W = self.W[0][self.W[1],:,:,:]
        else:
            W = self.W[0]
        if self.b[1] is not None:
            b = self.b[0][self.b[1]]
        else:
            b = self.b[0]
        rv = K.conv2d(X[:,:,:,:], W, border_mode='valid')  \
                 + K.reshape(b, (1, self.nb_domains, 1, 1))
        #return rv.dimshuffle((0,3,2,1))
        return rv #K.permute_dimensions(rv, (0,3,2,1))

"""
class OccConv(Layer):
    def __init__(self, num_bases, **kwargs, init='glorot_uniform'):
        self.num_bases = num_bases
        self.input = K.placeholder(ndim=4)
        self.init = lambda x: (
            initializations.get(init)(x), 
            K.zeros((self.nb_motifs,)) 
        )
        
        super(OccConv, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__, 
                  'num_bases': self.num_bases}
        base_config = super(OccConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def init_filters(self):
        self.W_shape = (
            self.nb_motifs, 4, 1, self.motif_len)
        self.W, self.b = self.init(self.W_shape)
        return

    @property
    def output_shape(self):
        #return self.input_shape
        assert self.input_shape[2] == 1
        return (
            self.input_shape[0],
            self.input_shape[1], # + (
            #    1 if self.input_shape[1]%self.num_tracks > 0 else 0),
            1,
            self.input_shape[3]-num_bases+1# + (
            #    1 if self.input_shape[3]%self.num_bases > 0 else 0)
        )

    def get_output(self, train=False):
        print "OccConv", self.output_shape
        X = self.get_input(train)
        X = K.permute_dimensions(X, (0,2,1,3))
        rv = K.conv2d(
            X, 
            pool_size=(num_tracks, num_bases), 
            strides=(num_tracks, num_bases),
            pool_mode='max'
        )
        return K.permute_dimensions(rv, (0,2,1,3))
"""


class LogAnyBoundOcc(Layer):
    def __init__(self, **kwargs):
        self.input = K.placeholder(ndim=4)        
        super(LogAnyBoundOcc, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(LogAnyBoundOcc, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output_shape(self):
        #return self.input_shape
        return (self.input_shape[0],
                self.input_shape[1],
                self.input_shape[2],
                1)

    def get_output(self, train=False):
        print "LogAnyBoundOcc", self.output_shape
        X = self.get_input(train)
        log_none_bnd = K.sum(
            K.log(1-K.clip(K.exp(X), 1e-6, 1-1e-6)), axis=3, keepdims=True)
        at_least_1_bnd = 1-K.exp(log_none_bnd)
        max_occ = K.max(K.exp(X), axis=3, keepdims=True)
        # we take the weighted sum because the max is easier to fit, and 
        # thus this helps to regularize the optimization procedure
        rv = K.log(0.05*max_occ + 0.95*at_least_1_bnd)
        return rv

custom_objects = {
  'ConvolutionDNASequenceBinding': ConvolutionDNASequenceBinding,
  'ConvolutionBindingSubDomains': ConvolutionBindingSubDomains,
  'LogNormalizedOccupancy': LogNormalizedOccupancy,
  'TrackMax': TrackMax,
  'OccMaxPool': OccMaxPool,
  'LogAnyBoundOcc': LogAnyBoundOcc,
  'ConvolutionCoOccupancy': ConvolutionCoOccupancy,
  'LogAnyBoundOcc': LogAnyBoundOcc,
  'LogNormalizedOccupancy': LogNormalizedOccupancy
}

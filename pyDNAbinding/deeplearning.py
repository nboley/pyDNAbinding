import os

import math
import hashlib
from itertools import chain
from collections import OrderedDict, defaultdict

import numpy as np
import h5py

class Data(object):
    """Store and iterate through data from a deep learning model.

    """
    def __len__(self):
        return len(self.inputs.values()[0])

    def _save_sequential(self, f):
        assert self._data_type == 'sequential'
        f.attrs['data_type'] = self._data_type
        inputs = f.create_dataset("inputs", data=self.inputs)
        outputs = f.create_dataset("outputs", data=self.outputs)
        task_ids = f.create_dataset("task_ids", data=self.task_ids)
        return

    def _save_graph(self, f):
        assert self._data_type == 'graph'
        f.attrs['data_type'] = self._data_type
        inputs = f.create_group("inputs")
        for key, val in self.inputs.iteritems():
            inputs.create_dataset(key, data=val)        
        outputs = f.create_group("outputs")
        for key, val in self.outputs.iteritems():
            outputs.create_dataset(key, data=val)
        task_ids = f.create_group("task_ids")
        for key, val in self.task_ids.iteritems():
            task_ids.create_dataset(key, data=val)
        return

    def __hash__(self):
        if self._cached_hash is not None:
            return self._cached_hash
        hashes = []
        if self._data_type == 'sequential':
            hashes.append(hashlib.sha1(self.input).hexdigest())
            hashes.append(hashlib.sha1(self.output).hexdigest())
            hashes.append(hash(self.task_ids))
        elif self._data_type == 'graph':
            hashes.append(hash(tuple(self.inputs.keys())))
            hashes.append(hash(tuple(self.outputs.keys())))
            hashes.append(hash(tuple(self.task_ids)))
            for val in self.inputs.values():
                # the array is required to be c contiguous to calculate the hash
                hashes.append(
                    hashlib.sha1(np.ascontiguousarray(val)).hexdigest())
            for val in self.outputs.values():
                # the array is required to be c contiguous to calculate the hash
                hashes.append(
                    hashlib.sha1(np.ascontiguousarray(val)).hexdigest())
        else:
            assert False, "Unrecognized data type '{}'".format(self._data_type)
        self._cached_hash = abs(hash(tuple(hashes)))
        return self._cached_hash

    @property
    def cache_fname(self):
        """File to write cached file into.
        
        This uses a hash over all of the stored data, so it should be completely
        safe to use. that is, if the underlying data changes then the hash value
        will change and the cached file will be regenerated.
        """
        return "cached%s.%i.obj" % (type(self).__name__, hash(self))
    
    def cache_to_disk(self):
        """Save a cached copy of this in an h5 file.

        Returns the cached filename.
        """
        if not os.path.isfile(self.cache_fname):
            self.save(self.cache_fname)
        return self.cache_fname
    
    def save(self, fname):
        """Save the data into an h5 file named fname.

        """
        with h5py.File(fname, "w") as f:
            if self._data_type == 'sequential':
                self._save_sequential(f)
            elif self._data_type == 'graph':
                self._save_graph(f)
            else:
                raise ValueError, "Unrecognized data type '{}'".format(
                    self._data_type)
    
    @classmethod
    def _load_sequential_data(cls, f):
        assert f.attrs['data_type'] == 'sequential'
        inputs = f['inputs']
        outputs = f['outputs']
        task_ids = f['task_ids']
        return inputs, outputs, task_ids
    
    @classmethod
    def _load_graph_data(cls, f):
        assert f.attrs['data_type'] == 'graph'
        inputs = f['inputs']
        outputs = f['outputs']
        task_ids = f['task_ids']
        return inputs, outputs, task_ids

    @classmethod
    def load(cls, fname):
        """Load data from an h5 file.

        Args:
            cls: type object of the return type
            fname: h5 file to load
        
        Returns:
            cls object with the inputs, outputs, and task_ids initialized to the
                values from the h5 file 'fname'
        
        Raises: 
            IOError: if fname isn't able to be read
        """
        print "Loading", fname
        f = h5py.File(fname, 'r')
        # This should probably also add a close method, but I there
        # would be very little purpose
        data_type = f.attrs['data_type']
        if data_type == 'sequential':
            inputs, outputs, task_ids = cls._load_sequential_data(f)
        elif data_type == 'graph':
            inputs, outputs, task_ids = cls._load_graph_data(f)
        else:
            raise ValueError,"Unrecognized data type '{}'".format(data_type)
        # we have to jump through some hoops to allow for proper subclassing. We
        # want the init methods to be able to allow for flexible argument 
        # patterns, but the interface all uses inputs, outputs, and taskids. So
        # we initialzie a class without calling init to get the correct type, 
        # and then we call __init__ on data directly.
        instance = cls.__new__(cls)
        Data.__init__(instance, inputs, outputs, task_ids)
        return instance
    
    def __init__(self, inputs, outputs, task_ids=None):
        self._cached_hash = None
        # if inputs is an array, then we assume that this is a sequential model
        if isinstance(inputs, (np.ndarray, h5py._hl.dataset.Dataset)):
            if not isinstance(outputs, (np.ndarray, h5py._hl.dataset.Dataset)):
                raise ValueError, "If the input is a numpy array, then the output is also expected to be a numpy array.\n" \
                    + "Hint: You can use multiple inputs and outputs by passing a dictionary keyed by the data type name."
            self._data_type = "sequential"
            self.num_observations = inputs.shape[0]
            assert self.num_observations == outputs.shape[0]
            # if no task ids are set, set them to indices
            if task_ids is not None:
                assert len(task_ids) == outputs.shape[1]
            else:
                task_ids = [str(x) for x in xrange(1, outputs.shape[1]+1)]
        # otherwise assume that this is a graph type model
        else:
            self.num_observations = inputs.values()[0].shape[0]
            # make sure that all of that data are arrays and have the same
            # number of observations
            for key, val in chain(inputs.iteritems(), outputs.iteritems()):
                assert isinstance(val, (np.ndarray, h5py._hl.dataset.Dataset))
                assert self.num_observations == val.shape[0], \
                    "The number of observations ({}) is not equal to the first shape dimension of {} ({})".format(
                        self.num_observations, key, val.shape[0])
            # make sure that task id length match up. If a task id key doesnt
            # exist, then default to sequential numbers
            if task_ids is None:
                task_ids = {}
            for key in outputs.iterkeys():
                if key in task_ids:
                    if len(task_ids[key]) != outputs[key].shape[1]:
                        raise ValueError, "The number of task ids for key '{}' does not match the output shape".format()
                else:
                    task_ids[key] = [
                        str(x) for x in xrange(1, outputs[key].shape[1]+1)]
            self._data_type = "graph"

        self.task_ids = task_ids
        self.inputs = inputs
        self.outputs = outputs
    
    def build_label_balanced_indices(self, label_key=None, task_id=None):
        # figure out the labels to sample from
        if self._data_type == "sequential":
            assert task_id is None
            labels = self.outputs
        else:
            if label_name is None and len(self.outputs) == 1:
                label_name = next(self.inputs.iterkeys())
            else:
                assert task_id is not None
                labels = self.inputs[task_id]

        if labels.shape[1] == 1:
            assert (task_id is None) or (task_id in self.task_ids), \
                "The specified task_id ({}) does not exist".format(task_id)
            labels = labels[:,0]
        else:
            assert task_id is not None, \
                "Must choose the task id to balance on in the multi-task setting"
            labels = labels[:,self.task_ids.index(task_id)]

        one_indices = np.random.choice(
            np.nonzero(labels == 1)[0], size=len(labels)/2)
        zero_indices = np.random.choice(
            np.nonzero(labels == 0)[0], size=len(labels)/2)
        permutation = np.concatenate((one_indices, zero_indices), axis=0)
        np.random.shuffle(permutation)
        return permutation

    def build_shuffled_indices(self):
        return np.random.permutation(self.num_observations)

    def build_ordered_indices(self):
        return np.arange(self.num_observations)

    def iter_batches_from_indices_generator(
            self, batch_size, repeat_forever, indices_generator):
        """XXX

        """
        i = 0
        n = int(math.ceil(self.num_observations/float(batch_size)))
        permutation = None
        while repeat_forever is True or i<n:
            if i%n == 0:
                permutation = indices_generator()
            # yield a subset of the data
            subset = slice((i%n)*batch_size, (i%n+1)*batch_size)
            indices = permutation[subset]
            # sort the indices because h5 indexing requires it
            indices.sort()
            if self._data_type == 'sequential':
                rv = (
                    self.inputs[indices.tolist()], 
                    self.outputs[indices.tolist()]
                )
            else:
                rv = {}
                for key, val in self.inputs.iteritems():
                    rv[key] = val[indices.tolist()]
                for key, val in self.outputs.iteritems():
                    rv[key] = val[indices.tolist()]
            yield rv
            i += 1
        return
    
    def iter_batches(self, 
                     batch_size, 
                     repeat_forever=False,
                     balanced=False, 
                     shuffled=False, 
                     **kwargs):
        if balanced:
            indices_generator = self.build_balanced_indices
        elif shuffled:
            indices_generator = self.build_shuffled_indices
        else:
            indices_generator = self.build_ordered_indices
        
        return self.iter_batches_from_indices_generator(
            batch_size, 
            repeat_forever, 
            indices_generator,
            **kwargs
        )

    def subset_observations(self, observation_indices=slice(None)):
        """Return a copy containing only the observations specified by indices

        indices: numpy array of indices to select
        """
        if self._data_type == 'sequential':
            assert isinstance(inputs, (np.ndarray, h5py._hl.dataset.Dataset))
            new_inputs = self.inputs[observation_indices]
            assert isinstance(outputs, (np.ndarray, h5py._hl.dataset.Dataset))
            new_outputs = self.outputs[observation_indices]
        elif self._data_type == 'graph':
            # indices must be sorted to subset an h5 array. This is fast, so we
            # jsut do it once at the start
            if isinstance(observation_indices, np.ndarray):
                observation_indices.sort()
            # subset the inputs
            new_inputs = {}
            for key, data in self.inputs.iteritems():
                if isinstance(data, np.ndarray):
                    new_inputs[key] = data[observation_indices]
                elif isinstance(data, h5py._hl.dataset.Dataset):
                    new_inputs[key] = data[observation_indices.tolist()]
                else:
                    raise ValueError, "Unrecognized data array type '%s'" \
                        % type(data)
            # subset the outputs
            new_outputs = {}
            for key, data in self.outputs.iteritems():
                if isinstance(data, np.ndarray):
                    new_outputs[key] = data[observation_indices]
                elif isinstance(data, h5py._hl.dataset.Dataset):
                    new_outputs[key] = data[observation_indices.tolist()]
                else:
                    raise ValueError, "Unrecognized data array type '%s'" \
                        % type(data)
        else:
            assert False,"Unrecognized model type '{}'".format(self._data_type)

        rv = Data(new_inputs, new_outputs, self.task_ids)
        rv.__class__ = self.__class__
        return rv
    
    def balance_data(self, label_key=None, task_id=None):
        indices = self.build_label_balanced_indices(
            label_key=label_key, task_id=task_id)
        return self.subset_observations(indices)
    
    def subset_tasks(self, desired_task_ids, label_key=None):
        """Return a copy of self that only contains desired_task_ids.

        FIX THIS FUNCTION
        XXX - DOCUMENT
        """
        # make chained filtering more convenient by defaulting to 
        # all tfs if none is passed
        if desired_task_ids is None:
            return self
        
        if label_key is None:
            if len(self.outputs) > 1:
                raise ValueError(
                    "If there are multiple output arrays then label_key must be specified.")
            else:
                label_key = self.outputs.keys()[0]
        
        new_outputs = {}
        # pass through the data for outputs not equal to label key
        for task_key, data in self.outputs.iteritems():
            if task_key != label_key:
                new_outputs[task_key] = data
            
        task_indices = []
        for task_id in desired_task_ids[label_key]:
            try: task_indices.append(self.task_ids[label_key].index(task_id))
            except ValueError: task_indices.append(-1)
        task_indices = np.array(task_indices)

        new_data = np.insert(data, 0, -1, axis=1)
        new_outputs[label_key] = new_data[:, task_indices+1]

        rv = Data(self.inputs, new_outputs, task_ids=desired_task_ids)
        rv.__class__ = self.__class__
        return rv

class GenomicRegionsAndLabels(Data):
    """Subclass Data to handfle the common case where the input is a set of 
       genomic regions and the output is a single labels matrix. 
    
    """    
    def __len__(self):
        return len(self.regions)
    
    @property
    def regions(self):
        return self.inputs['regions']

    def _subset_regions_by_rank(
            self, max_num_regions, use_top_regions=False, seed=0):
        """Return a copy of self containing at most max_num_regions regions.

        max_num_regions: the maximum number of regions to return
        use_top_accessible: return max_num_regions most accessible regions
        """
        if max_num_regions is None:
            max_num_regions = len(self.regions)

        # set a seed so we can use cached regions between debug rounds
        np.random.seed(seed)
        
        # sort the regions by accessibility
        if use_top_regions:
            if 'signalValue' not in self.regions:
                raise ValueError(
                    "Can not subset top regions unless a signal value is provided")
            # sort indices by signal value, breaking ties using a random sort
            indices = np.lexsort(
                (np.random.random(len(self.regions)), 
                 -self.regions['signalValue'])
            )
        # sort the regions randomly
        else:
            indices = np.argsort(np.random.random(len(self.regions)))
        
        return self.subset_observations(indices[:max_num_regions])

    def subset_random_regions(self, max_num_regions, seed=0):
        return self._subset_regions_by_rank(
            max_num_regions, use_top_regions=False, seed=seed)
    
    def subset_top_regions(self, max_num_regions, seed=0):
        return self._subset_regions_by_rank(
            max_num_regions, use_top_regions=True, seed=seed)
    
    def subset_regions_by_contig(
            self, contigs_to_include=None, contigs_to_exclude=None):
        assert (contigs_to_include is None) != (contigs_to_exclude is None), \
            "Either contigs_to_include or contigs_to_exclude must be specified"
        # case these to sets to speed up the search a little
        if contigs_to_include is not None:
            contigs_to_include = set(contigs_to_include)
        if contigs_to_exclude is not None:
            contigs_to_exclude = set(contigs_to_exclude)
        indices = np.array([
            i for i, region in enumerate(self.regions) 
            if (contigs_to_exclude is None or region[0] not in contigs_to_exclude)
            and (contigs_to_include is None or region[0] in contigs_to_include)
        ])
        return self.subset_observations(indices)
    
    def __init__(self, regions, labels, inputs={}, task_ids=None):
        # add regions to the input
        if 'regions' in inputs:
            raise ValueError(
                "'regions' input is passed as an argument and also specified in inputs")
        inputs['regions'] = regions
        Data.__init__(self, inputs, {'labels': labels}, {'labels': task_ids})

class SamplePartitionedData():
    """Store data partitioned by sample id.

    """
    def __len__(self):
        return sum(len(x) for x in self._data.itervalues())

    @property
    def sample_ids(self):
        return self._data.keys()

    def __hash__(self):
        if self._cached_hash is not None:
            return self._cached_hash
        hashes = [hash(tuple(sorted(self._data.keys()))),
        ] + [hash(val) for val in self._data.values()]
        self._cached_hash = abs(hash(tuple(hashes)))
        return self._cached_hash

    @property
    def cache_fname(self):
        return "cached%s.%i.obj" % (type(self).__name__, hash(self))

    def cache_to_disk(self):
        """Save a cached copy of this in an h5 file.

        Returns the cached filename.
        """
        if not os.path.isfile(self.cache_fname):
            self.save(self.cache_fname)
        return self.cache_fname
    
    def save(self, fname):
        with h5py.File(fname, "w") as f:
            for key, data in self._data.iteritems():
                sample_fname = data.cache_to_disk()
                f[key] = h5py.ExternalLink(sample_fname, "/")
        return fname

    @classmethod
    def load(cls, fname):
        """Load data from an h5 file.

        Args:
            cls: type object of the return type
            fname: h5 file to load
        
        Returns:
            cls object with the inputs, outputs, and task_ids initialized to the
                values from the h5 file 'fname'
        
        Raises: 
            IOError: if fname isn't able to be read
        """
        rv  = {}
        with h5py.File(fname, 'r') as f:
            for key, data in f.iteritems():
                rv[key] = Data(**data)
        return cls(rv)

    def iter_batches(self, 
                     batch_size, 
                     repeat_forever=False, 
                     sample_weights=None, 
                     **kwargs):
        """Iterate data in batches. 

        SamplePartitionedData is a container object that stores samples and 
        data. This method calls of each of the data iterators, and adds the 
        proper sample id into the returned data dictionary.
        
        Args:
            batch_size: size of the first dimension of the returned data.
            repeat_forever: if set then continuously iterate through the data.
            **kwargs: additional arguments to pass to the sub iterators
        
        Returns:
            cls object with the inputs, outputs, and task_ids initialized to the
                values from the h5 file 'fname'
        
        Raises: 
            IOError: if fname isn't able to be read
        """

        ## find the number of observations to sample from each batch
        # if sample weights is not set, then use equal weights
        if sample_weights is None:
            sample_weights = np.ones(len(self._data), dtype=float)
        fractions = sample_weights*np.array([
            x.num_observations for x in self._data.values()], dtype=float)
        fractions = fractions/fractions.sum()
        inner_batch_sizes = np.array(batch_size*fractions, dtype=int)
        # accounting for any rounding from the previous step 
        for i in xrange(batch_size - inner_batch_sizes.sum()):
            inner_batch_sizes[i] += 1

        iterators = OrderedDict(
            (sample_id, 
             data.iter_batches(
                 i_batch_size, repeat_forever, **kwargs) )
            for i_batch_size, (sample_id, data) in zip(
                    inner_batch_sizes, self._data.iteritems())
        )

        def f():
            while True:
                grpd_res = defaultdict(list)
                cnts = []
                for sample_id in self.sample_ids:
                    if sample_id not in iterators:
                        cnts.append(0)
                    else:
                        assert sample_id in iterators
                        iterator = iterators[sample_id]
                        data = next(iterator)
                        cnt = None
                        for key, vals in data.iteritems():
                            grpd_res[key].append(vals)
                            if cnt == None: cnt = vals.shape[0]
                            assert cnt == vals.shape[0]
                        cnts.append(cnt)
                
                for key, vals in grpd_res.iteritems():
                    grpd_res[key] = np.concatenate(grpd_res[key], axis=0)
                    
                # build the sample labels
                cnts = np.array(cnts)
                sample_labels = np.zeros(
                    (cnts.sum(), len(self.sample_ids)), dtype='float32')
                for i in xrange(len(cnts)):
                    start_index = (0 if i == 0 else np.cumsum(cnts)[i-1])
                    stop_index = np.cumsum(cnts)[i]
                    sample_labels[start_index:stop_index,i] = 1                
                assert 'sample_ids' not in grpd_res
                grpd_res['sample_ids'] = sample_labels
                
                # cast this to a normal dict (rather than a default dict)
                yield dict(grpd_res)
            return
        
        return f()

    def __init__(self, samples_and_data):
        self._cached_hash = None
        try: 
            self._data = OrderedDict(samples_and_data.iteritems())
        except AttributeError:
            self._data = OrderedDict(samples_and_data)
        for val in self._data.itervalues():
            assert isinstance(val, Data)
        return

def test_load_and_save(inputs, outputs):
    data = Data(inputs, outputs)
    data.save("tmp.h5")    
    data2 = Data.load("tmp.h5")
    for x in data2.iter_batches(10):
        break
    return

def test_hash():
    s1 = Data({'seqs': np.zeros((10000, 50))}, {'labels': np.zeros((10000, 1))})
    s2 = Data({'seqs': np.zeros((10000, 50))}, {'labels': np.zeros((10000, 1))})
    assert hash(s1) == hash(s2)

def test_sample_partitioned_data():
    s1 = Data({'seqs': np.zeros((10000, 50))}, {'labels': np.zeros((10000, 1))})
    s2 = Data({'seqs': np.zeros((10000, 50))}, {'labels': np.zeros((10000, 1))})
    s = SamplePartitionedData({'s1': s1, 's2': s2})
    fname = s.cache_to_disk()
    s2 = SamplePartitionedData.load(fname)
    for x in s2.iter_batches(10):
        for key, val in x.iteritems():
            print key, val.shape
            print val
        break
    return

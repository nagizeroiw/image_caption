import time
import cPickle
import numpy

from config import Config


def load_pkl(path):
    """Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval


def dump_pkl(obj, path):
    """Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


def generate_minibatch_idx(dataset_size, minibatch_size):
    """Generate idx for minibatches SGD.

    Output:
        A list of [m1, m2, m3, ..., mk],  where mk is a list of indices
    """
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
    else:
        print 'uneven minibatch chunking, overall %d, last one %d' % \
            (minibatch_size, leftover)
        minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx


class DataEngine():

    def __init__(self):

        dataset_root = Config.dataset_dir
        print '>>> loading image_caption dataset...'
        t = time.time()

        # caption
        self.cap = load_pkl(dataset_root + 'caps.pkl')

        # image feature of shape (1, 1536)
        self.feature = load_pkl(dataset_root + 'feature.pkl')

        # vid-cid pairs for train/valid/test.
        self.train = load_pkl(dataset_root + 'train.pkl')
        self.valid = load_pkl(dataset_root + 'valid.pkl')

        # vids for trian/valid/test, coresponding with elements above.
        self.train_ids = load_pkl(dataset_root + 'train_vids.pkl')
        self.valid_ids = load_pkl(dataset_root + 'valid_vids.pkl')

        # saved vid2name
        self.vid2name = load_pkl(dataset_root + 'vid2name.pkl')

        # worddict: word -> id
        self.worddict = load_pkl(dataset_root + 'worddict.pkl')
        self.word_idict = dict()
        for word, id in self.worddict.iteritems():
            self.word_idict[id] = word
        # add special words
        self.worddict['<eos>'] = 0
        self.worddict['<UNK>'] = 1
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = '<UNK>'

        # generate indices for SGD
        self.kf_train = generate_minibatch_idx(len(self.train), Config.batch_size)
        self.kf_valid = generate_minibatch_idx(len(self.valid), Config.batch_size)

        print '>>> image_caption dataset loaded. [time %.2f]' % (time.time() - t)

    def iter_valid_image_features(self):
        """return iteration of all valid image features.

        Output:
            (image_name: str, image_feature: (1536,))
        """
        for id in self.valid_ids:
            name = self.vid2name[id].split('/')[-1]  # './train/.../dsakfsda.jpg' -> 'dsakfsda.jpg'
            name = name.split('.')[0]  # 'dsakfsda'
            yield (name, self.feature[id][0, :])

    def prepare_data(self, IDs):
        seqs = []
        feat_list = []

        def get_words(imgID, capID):
            caps = self.cap[imgID]
            rval = None
            for cap in caps:
                if str(cap['cap_id']) == capID:
                    rval = cap['tokenized'].split(' ')
                    break
            assert rval is not None
            return rval

        for i, ID in enumerate(IDs):
            # print 'processed %d/%d caps' % (i, len(IDs))
            imgID, capID = ID.split('_')

            words = get_words(imgID, capID)

            seq = []
            for w in words:
                seq.append(self.worddict.get(w, self.worddict['<UNK>']))

            # only short sentences are used.
            if len(seq) >= Config.maxlen:
                continue
            seqs.append(seq)

            # feature (1, 1536) -> (1536,)
            feat = self.feature[imgID][0, :]
            feat_list.append(feat)

        n_samples = len(seqs)
        if n_samples == 0:
            return None, None, None

        # as demanded by pack_padded_sequence,
        # feature and seq should be sorted according to length.
        lengths = [len(s) for s in seqs]
        length = numpy.asarray(lengths).astype('int64')

        # get the indices of the indirect sort.
        indices = numpy.argsort(-length)

        # get sorted length # ?
        length = length[indices]

        # get sorted feature
        ordered_feat_list = []
        for i in range(n_samples):
            ordered_feat_list.append(feat_list[indices[i]])
        feature = numpy.asarray(ordered_feat_list).astype('float32')

        seq = numpy.zeros((n_samples, Config.maxlen)).astype('int64')
        for idx in range(n_samples):
            seq[idx, :lengths[indices[idx]]] = seqs[indices[idx]]
            seq[idx, lengths[indices[idx]]] = self.worddict['<eos>']

        length += 1

        # assert (feature[0] == feat_list[indices[0]]).all()
        return feature, seq, length


def test_data_engine():

    print '>>> test_data_engine() ...'
    t = time.time()

    engine = DataEngine()

    i = 0
    # load first 10 batches from train
    for idx in engine.kf_train:
        t0 = time.time()
        i += 1
        ids = [engine.train[index] for index in idx]
        feature, seq, length = engine.prepare_data(ids)

        print ' seen %d batches, time used %.2f seconds.' % (i, time.time() - t0)

        if i == 9:
            print ' feature.shape, seq.shape, length.shape', feature.shape, seq.shape, length.shape
            print ' examine seq'
            print length[:5]
            print seq[0, :]
            print u' '.join([engine.word_idict[x] for x in seq[0, :length[0]]])
            break

    print '>>> test_data_engine() ended. [time %.2f]' % (time.time() - t)


if __name__ == '__main__':
    test_data_engine()

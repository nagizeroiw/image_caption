import time
import datetime
import os
import random
import json
import io

import torch
import torch.nn as nn
from torch.autograd import Variable

import progressbar

from model import Caption
from data_engine import DataEngine
from config import Config


def main(which_set='valid'):

    # init data engine
    data_engine = DataEngine()

    # init model
    model = Caption()
    if Config.use_cuda:
        model.cuda()

    ckpt_name = Config.infer_ckpt_name if which_set == 'valid' else Config.test_ckpt_name

    # loaded trained model parameters
    print '>>> Loading checkpoint...'
    if os.path.isfile(ckpt_name):
        checkpoint = torch.load(ckpt_name)
        print ' checkpoint found: [%s]' % (checkpoint['time'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print ' checkpoint not found at %s.' % ckpt_name
        return

    print '>>> Start inferencing...'
    t = time.time()

    result = []
    i = 0
    bar = progressbar.ProgressBar(maxval=len(data_engine.valid_ids))
    bar.start()

    generator = data_engine.iter_valid_image_features \
        if which_set == 'valid' else data_engine.iter_test_image_features

    for name, feature in generator():

        feature = Variable(torch.from_numpy(feature))
        feature = feature.cuda() if Config.use_cuda else feature

        # get the top sentence of beam_search
        sentence = model.inference(feature)

        # make the sentence ALIVE!
        sentence = u''.join([data_engine.word_idict[w] for w in sentence])

        i += 1
        cap = {}
        cap['image_id'] = name
        cap['caption'] = sentence
        result.append(cap)

        bar.update(i)

    bar.finish()

    file_name = Config.inference_file if which_set == 'valid' else Config.test_inference_file

    print '>>> Saving inference results to %s...' % (file_name)

    # end
    result_json = json.dumps(result, ensure_ascii=False)
    with io.open(file_name, 'w', encoding='utf-8') as file:
        file.write(result_json)

    print '>>> end inferencing. [time %.2f]' % (time.time() - t)


if __name__ == '__main__':

    main(which_set='test')

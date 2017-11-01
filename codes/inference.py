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


def main():

    # init data engine
    data_engine = DataEngine()

    # init model
    model = Caption()
    if Config.use_cuda:
        model.cuda()

    # loaded trained model parameters
    print '>>> Loading checkpoint...'
    if os.path.isfile(Config.train_ckpt_name):
        checkpoint = torch.load(Config.train_ckpt_name)
        print ' checkpoint found: [%s]' % (checkpoint['time'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print ' checkpoint not found at %s.' % Config.train_ckpt_name
        return

    print '>>> Start inferencing...'
    t = time.time()

    result = []
    i = 0
    bar = progressbar.ProgressBar(maxval=len(data_engine.valid_ids))
    bar.start()

    for name, feature in data_engine.iter_valid_image_features():

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

    print '>>> Saving inference results to %s...' % (Config.inference_file)

    # end
    result_json = json.dumps(result, ensure_ascii=False)
    with io.open(Config.inference_file, 'w', encoding='utf-8') as file:
        file.write(result_json)

    print '>>> end inferencing. [time %.2f]' % (time.time() - t)


if __name__ == '__main__':

    main()

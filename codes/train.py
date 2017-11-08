import time
import datetime
import os
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR

from model import Caption
from data_engine import DataEngine
from config import Config
import plot
import inference

from peval.run_evaluations import compute_m1


def main():

    # init data_engine, load dataset
    data_engine = DataEngine()

    # init model
    model = Caption()

    if Config.use_cuda:
        model.cuda()

    # update counts
    uid = 0

    # start epoch
    start_epoch = 0

    # saved loss (for computing average)
    saved_loss = 0.
    saved_valid_loss = 0.

    # for plot
    iters, losses = [], []
    valid_iters, valid_losses = [], []
    # for eval
    eval_iters, eval_m, eval_b = [], [], []

    # clear eval result file
    with open(Config.eval_file, 'w') as file:
        file.write('  update  B-4    C    M    R\n')

    if Config.is_reload:
        print '>>> Loading checkpoint...'
        if os.path.isfile(Config.train_ckpt_name):

            checkpoint = torch.load(Config.train_ckpt_name)

            print ' checkpoint found: [%s]' % (checkpoint['time'])
            uid = checkpoint['uid'] if 'uid' in checkpoint else uid
            start_epoch = checkpoint['start_epoch'] if 'start_epoch' in checkpoint else start_epoch
            iters = checkpoint['iters'] if 'iters' in checkpoint else iters
            losses = checkpoint['losses'] if 'losses' in checkpoint else losses

            valid_iters = checkpoint['valid_iters'] \
                if 'valid_iters' in checkpoint else valid_iters
            valid_losses = checkpoint['valid_losses'] \
                if 'valid_losses' in checkpoint else valid_losses

            model.load_state_dict(checkpoint['state_dict'])

        else:
            print ' checkpoint not found at %s.' % Config.train_ckpt_name

    # training tools
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=Config.learning_rate, weight_decay=1e-4)

    # last_epoch?
    # last_epoch = -1 if start_epoch == 0 else start_epoch
    # scheduler = StepLR(optimizer, step_size=8, gamma=0.5, last_epoch=last_epoch)

    print '>>> Start training...'
    t = time.time()

    for epoch in range(Config.n_epoch):

        # adjust learning rate
        # scheduler.step()

        for i_epo, idx in enumerate(data_engine.kf_train):

            ids = [data_engine.train[id] for id in idx]
            uid += 1

            # get data of this mini-batch
            features, seqs, lengths = data_engine.prepare_data(ids)
            if features is None:
                continue
            features = Variable(torch.from_numpy(features))
            seqs = Variable(torch.from_numpy(seqs))

            if Config.use_cuda:
                features, seqs = features.cuda(), seqs.cuda()

            # pack targets for computing loss
            targets = pack_padded_sequence(seqs, lengths, batch_first=True)[0]

            # forward, backward, and optimization step
            model.zero_grad()
            outputs = model(features, seqs, lengths)
            # print outputs.shape

            loss = criterion(outputs, targets)

            # normalization
            max_length = max(lengths)
            loss.data[0] /= max_length
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), Config.gradient_clip)
            optimizer.step()

            saved_loss += loss.data[0]

            # print info
            if uid % Config.train_log_freq == 0 or Config.is_debug:
                print '[Training] Epoch: %3d(%2d%%) Update: %8d Loss: %8.5f' \
                    % (epoch + start_epoch, int(i_epo * 100. / len(data_engine.kf_train)),
                       uid, loss.data[0])

            # save info for visualization
            if uid % Config.train_plot_freq == 0 or Config.is_debug:
                iters.append(uid)
                losses.append(saved_loss / Config.train_plot_freq)
                saved_loss = 0

            # validation
            if uid % Config.valid_freq == 0:

                # randomly choose a batch of data to compute loss on valid set
                idx = random.choice(data_engine.kf_valid)
                ids = [data_engine.valid[id] for id in idx]

                features, seqs, lengths = data_engine.prepare_data(ids)
                features = Variable(torch.from_numpy(features))
                seqs = Variable(torch.from_numpy(seqs))

                if Config.use_cuda:
                    features, seqs = features.cuda(), seqs.cuda()

                targets = pack_padded_sequence(seqs, lengths, batch_first=True)[0]

                # forward, and compute loss
                model.zero_grad()
                outputs = model(features, seqs, lengths)
                loss = criterion(outputs, targets)

                # normalization
                max_length = max(lengths)
                loss.data[0] /= max_length
                saved_valid_loss += loss.data[0]

            # print and save valid loss
            if uid % (Config.valid_plot_freq) == 0:
                num = Config.valid_plot_freq / Config.valid_freq
                valid_iters.append(uid)
                print '[Validation] Loss: %8.5f' % (saved_valid_loss / num)
                valid_losses.append(saved_valid_loss / num)
                saved_valid_loss = 0

            # sampling
            if uid % Config.sample_freq == 0 or Config.is_debug:

                def sampling(name, vid_set, kf_set):

                    # sampling from train
                    print '[Sampling] Sampling from %s set...' % name
                    t0 = time.time()

                    for _ in range(Config.sample_count):
                        idx = random.choice(kf_set)
                        ids = [vid_set[id] for id in idx]
                        features, seqs, lengths = data_engine.prepare_data(ids)

                        # randomly select one image for sampling
                        q = random.randint(0, len(lengths) - 1)
                        feature, seq, length = features[q], seqs[q], lengths[q]
                        feature = Variable(torch.from_numpy(feature))
                        feature = feature.cuda() if Config.use_cuda else feature

                        # beam search sampling
                        captions, scores = model.beam_search(feature)

                        # print ground truth
                        print ' > id: %s' % ids[q]
                        print '  ground truth:', \
                            u''.join([data_engine.word_idict[w] for w in seq[:length]])

                        # print model outputs
                        for i in range(min(3, len(scores))):
                            print '  output #%d (%6.4f):' % (i, scores[i]), \
                                u''.join([data_engine.word_idict[w] for w in captions[i]])

                    print '[Sampling] Sampling ended. Average time: %4.2fs' \
                        % ((time.time() - t0) / Config.sample_count)

                sampling('train', data_engine.train, data_engine.kf_train)
                sampling('valid', data_engine.valid, data_engine.kf_valid)

            # saving loss curve
            if uid % Config.plot_freq == 0:

                print '[Visualization] Saving loss curve to %s.' % (Config.loss_plot_file)
                plot.compare_xy(iters, losses, valid_iters, valid_losses, Config.loss_plot_file)

            # inference
            if uid % Config.check_freq == 0:

                print '[Inference] check model performance'

                print '>>> Saving checkpoint uid%d to %s.' % (uid, Config.train_ckpt_name)
                torch.save({'state_dict': model.state_dict(),
                            'uid': uid,
                            'start_epoch': start_epoch + Config.n_epoch,
                            'iters': iters,
                            'losses': losses,
                            'valid_iters': valid_iters,
                            'valid_losses': valid_losses,
                            'time': str(datetime.datetime.now())},
                           Config.train_ckpt_name)

                # start inferencing
                inference.main()

                # evaluation
                m1_score = compute_m1(Config.inference_file,
                                      '/home/kesu/image_caption/dataset/valid_reference.json')

                # show and save results
                print m1_score
                eval_iters.append(uid)
                eval_m.append(m1_score['METEOR'])
                eval_b.append(m1_score['Bleu_4'])

                # save best model
                if m1_score['Bleu_4'] >= max(eval_b):

                    print '>>> Saving best performance checkpoint uid%d to %s.' \
                        % (uid, Config.best_ckpt_file)
                    torch.save({'state_dict': model.state_dict(),
                                'uid': uid,
                                'start_epoch': start_epoch + Config.n_epoch,
                                'iters': iters,
                                'losses': losses,
                                'valid_iters': valid_iters,
                                'valid_losses': valid_losses,
                                'time': str(datetime.datetime.now())},
                               Config.best_ckpt_file)

                # save results to file
                with open(Config.eval_file, 'a') as file:
                    file.write('%8d %4f %4f %4f %4f\n' % (uid, m1_score['Bleu_4'], m1_score['CIDEr'],
                                                          m1_score['METEOR'], m1_score['ROUGE_L']))

                # save results
                print '>>> Saving visualized METEOR curve to %s.' % (Config.eval_plot_file)
                plot.compare_xy(eval_iters, eval_m, eval_iters, eval_b,
                                Config.eval_plot_file, 'M', 'B4')

                print '[Inference] inference done.'

            # only update once if debugging
            if Config.is_debug or uid > Config.n_updates:
                break

        # quit if debugging
        if Config.is_debug or uid > Config.n_updates:
            break

    if Config.is_debug:
        print '>>> Debug end. [time %.2f]' % (time.time() - t)
        return

    # print '>>> saving visualized training process to %s.' % (Config.result_dir)
    # plot.show_xy(iters, losses, Config.result_dir + 'training.png')

    # print '>>> saving visualized validating process to %s.' % (Config.result_dir)
    # plot.show_xy(valid_iters, valid_losses, Config.result_dir + 'validating.png')

    print '>>> Saving checkpoint uid%d to %s.' % (uid, Config.ckpt_dir)
    torch.save({'state_dict': model.state_dict(),
                'uid': uid,
                'start_epoch': start_epoch + Config.n_epoch,
                'iters': iters,
                'losses': losses,
                'valid_iters': valid_iters,
                'valid_losses': valid_losses,
                'time': str(datetime.datetime.now())},
               Config.train_ckpt_name)

    print '>>> End training. [time %.2f]' % (time.time() - t)


if __name__ == '__main__':

    main()

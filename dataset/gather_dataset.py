'''

    This code gathers the dataset from the AI Challenge dataset
    to the demanded format.

    Ke su
    2017/10/11

'''

import cPickle
import jieba
import json
import io
import sys
import hashlib

import progressbar


caps_name = 'caps.pkl'
vid2name_name = 'vid2name.pkl'
train_vids_name = 'train_vids.pkl'
valid_vids_name = 'valid_vids.pkl'
train_name = 'train.pkl'
valid_name = 'valid.pkl'
worddict_name = 'worddict.pkl'


def load_pkl(pkl_file_name):
    with open(pkl_file_name, 'rb') as pkl_file:
        return cPickle.load(pkl_file)


def dump_pkl(obj, pkl_file_name):
    with open(pkl_file_name, 'wb') as pkl_file:
        return cPickle.dump(obj, pkl_file, protocol=cPickle.HIGHEST_PROTOCOL)


def load_test_set():
    test_dir = './image_caption_data/test1/'
    test_images_dir = test_dir + 'caption_test1_images_20170923/'

    # list of 'vid100002'
    test_vids = []
    # dict of 'vid100002' -> './test1/***images**/3905ofjoawienf.jpg'
    vid2name_test = {}


def load_dataset():

    '''
    ignore test (inference) step currently
    get:
        [train/valid]_vids: two lists of video IDs like 'vid1'
        vid2name: dictionary='vid1' -> './train/***images**/3905ofjoawienf.jpg'
            contains both train and valid vIDs
        caps: dictionary='vid1' -> [{'tokenized':'for train',
                                     'caption':'for caption',
                                     'image_id':'vid1', 'cap_id':'0'},
                                    {''}, ...]
        train/valid/test: lists of VID-CAPID pairs like 'vid1_0'

        worddict: dictionary='boy' -> 5
    '''

    # dir to caption json files and image folders
    train_dir = './image_caption_data/train/'
    valid_dir = './image_caption_data/valid/'

    train_caption_dir = train_dir + 'caption_train_annotations_20170902.json'
    train_images_dir = train_dir + 'caption_train_images_20170902/'

    valid_caption_dir = valid_dir + \
        'caption_validation_annotations_20170910.json'
    valid_images_dir = valid_dir + \
        'caption_validation_images_20170910/'

    # [train/valid]_vids: two lists of video IDs like 'vid1'
    #   vid2name: dictionary='vid1' -> './train/***images**/3905ofjoawienf.jpg'
    #     contains both train and valid vIDs
    train_vids = []
    valid_vids = []
    vid2name = {}

    # caps: dictionary='vid1' -> [{'tokenized':'for train',
    #                              'caption':'for eval',
    #                              'image_id':'vid1',
    #                              'cap_id':'0'},
    #                             {''}, ...]
    caps = {}

    # train/valid/test: lists of VID-CAPID pairs like 'vid1_0'
    train = []
    valid = []
    # test = []

    # worddict: dictionary='boy' -> 5
    # wordcount: dictionary='boy' -> 384
    worddict = {}
    wordcount = {}

    # current parsing vid
    known_vid = 0

    print '>>> reading AIC dataset...'

    def parse_dataset(caption_dir, image_dir, vids_list, pairs_list, current_vid):

        with open(caption_dir, 'r') as caption_file:
            caption_content = json.load(caption_file)
            for image in caption_content:

                #  add to dataset
                current_vid += 1
                this_vid = 'vid%d' % current_vid
                # maintain train_vids
                vids_list.append(this_vid)

                #  get information
                # url = image['url'] redundant
                image_name = image['image_id']
                # maintain vid2name
                vid2name[this_vid] = image_dir + image_name
                caption_list = image['caption']

                #  build cap_list for this image
                cap_list = []
                for i, caption in enumerate(caption_list):
                    # cut caption to list
                    # TODO: caption should be concatenate to string later
                    seg_list = jieba.cut(caption, cut_all=False)
                    seg_list = [word for word in seg_list]

                    # maintain wordcount
                    for word in seg_list:
                        if word in wordcount:
                            wordcount[word] += 1
                        else:
                            wordcount[word] = 1

                    # build cap dictionary
                    cap = {}
                    cap['tokenized'] = seg_list
                    cap['caption'] = seg_list
                    cap['image_id'] = current_vid
                    cap['cap_id'] = str(i)
                    cap_list.append(cap)

                    # maintain train
                    pairs_list.append(this_vid + '_' + str(i))

                # maintain caps
                caps[this_vid] = cap_list

        return current_vid

    # parse train and valid dataset
    known_vid = parse_dataset(train_caption_dir, train_images_dir, train_vids, train, current_vid=known_vid)
    parse_dataset(valid_caption_dir, valid_images_dir, valid_vids, valid, current_vid=known_vid)

    # maintain worddict
    #  item: (word, count)
    wordlist = [item for item in wordcount.items() if item[1] >= 5]
    wordlist = sorted(wordlist, key=lambda x: x[1], reverse=True)
    wordlist = [word for word, count in wordlist]
    for i, word in enumerate(wordlist):
        worddict[word] = i + 5

    # re-parse caps dictionary (concatenate caption)
    for vid, caption_list in caps.iteritems():
        for caption in caption_list:

            # when training, ignore rare words
            caption_for_train = ' '.join(
                [word for word in caption['tokenized'] if word in worddict])
            caption_for_eval = ' '.join(caption['tokenized'])

            # for train
            caption['tokenized'] = caption_for_train

            # for eval
            caption['caption'] = caption_for_eval

    # display statistic results
    print '>>> display statistic results:'

    print ' >>> caps size:', len(caps)
    print ' >>> train_vids size, valid_vids size:', \
        len(train_vids), len(valid_vids)

    print ' >>> total captions:', sum(
        [len(caplist) for caplist in caps.values()])
    print ' >>> train size, valid size:', len(train), len(valid)

    print ' >>> top words:', u' '.join(wordlist[:50])
    print ' >>> worddict size:', len(worddict)

    print ' >>> tokenized sample:', caps['vid1'][0]['tokenized']
    print ' >>> caption sample', caps['vid1'][0]['caption']

    # dump results
    print '>>> dump results'
    dump_pkl(caps, caps_name)
    dump_pkl(vid2name, vid2name_name)
    dump_pkl(train_vids, train_vids_name)
    dump_pkl(valid_vids, valid_vids_name)
    dump_pkl(train, train_name)
    dump_pkl(valid, valid_name)
    dump_pkl(worddict, worddict_name)


def look_dataset():

    print '>>> loading dataset...'
    caps = load_pkl(caps_name)
    vid2name = load_pkl(vid2name_name)
    train_vids = load_pkl(train_vids_name)
    valid_vids = load_pkl(valid_vids_name)
    train = load_pkl(train_name)
    valid = load_pkl(valid_name)
    worddict = load_pkl(worddict_name)

    # display statistic results
    print '>>> display statistic results:'

    print ' >>> vid2name samples:', '%s -> %s' % ('vid2', vid2name['vid2'])

    print ' >>> caps size:', len(caps)
    print ' >>> train_vids size, valid_vids size:', \
        len(train_vids), len(valid_vids)

    print ' >>> total captions:', sum(
        [len(caplist) for caplist in caps.values()])
    print ' >>> train size, valid size:', len(train), len(valid)

    # print ' >>> top words:', u' '.join(wordlist[:50])
    print ' >>> worddict size:', len(worddict)

    print ' >>> tokenized sample:', caps['vid1'][0]['tokenized']
    print ' >>> caption sample', caps['vid1'][0]['caption']


def build_valid_reference_json():

    print '>>> loading dataset...'

    # caps: dictionary='vid1' -> [{'tokenized':'for train',
    #                              'caption':'for eval',
    #                              'image_id':'vid1',
    #                              'cap_id':'0'},
    #                             {''}, ...]
    caps = load_pkl(caps_name)
    vid2name = load_pkl(vid2name_name)
    valid_vids = load_pkl(valid_vids_name)

    # image names and ids.
    '''
          "images": [
            {
              "file_name": "d9a9a8cfb9fdebe3f4f6996756ae23ae0e473d0c",
              "id": 6678090646845985471
            }, ...]
    '''
    images = []

    # (cutted) captions, cap_ids, and corresponding image ids.
    '''
          "annotations": [
            {
              "caption": "\u4e00\u4e2a \u957f \u5934\u53d1 \u7684 \u5973\u4eba
                \u5750\u5728 \u6d77\u8fb9 \u6c99\u6ee9 \u7684 \u6905\u5b50 \u4e0a
                \u770b\u7740 \u84dd\u5929 \u5927\u6d77",
              "id": 1,
              "image_id": 6678090646845985471
            }, ...]
    '''
    annotations = []

    current_caption_id = 1
    current_image_id = 1

    bar = progressbar.ProgressBar(maxval=len(valid_vids))
    bar.start()

    for id in valid_vids:
        captions = caps[id]

        bar.update(current_image_id)

        image = {}
        name = vid2name[id].split('/')[-1]
        name = name.split('.')[0]
        image['file_name'] = name
        image['id'] = int(int(hashlib.sha256(name).hexdigest(), 16) % sys.maxint)
        current_image_id += 1

        images.append(image)

        for caption in captions:
            cap = {}
            cap_list = caption['caption'].split(' ')
            cap['caption'] = ' '.join([x for x in cap_list])  # repr(x)[2:-1] for x in cap_list])
            cap['id'] = current_caption_id
            cap['image_id'] = image['id']

            current_caption_id += 1
            annotations.append(cap)

    bar.finish()

    result = {}
    result['annotations'] = annotations
    result['images'] = images
    result['type'] = 'captions'
    result['info'] = {
        'contributor': 'Ke Su',
        'description': 'CaptionEval',
        'url': 's',
        'version': '1',
        'year': '2017'
    }
    result['licenses'] = [
        {
            'url': 's'
        }
    ]

    print result['annotations'][0]['caption']

    output_file_name = 'valid_reference.json'
    result_json = json.dumps(result, ensure_ascii=False)
    # result_json = result_json.replace('\\\\', '\\')
    with io.open(output_file_name, 'w', encoding='utf-8') as file:
        file.write(result_json)

    print '>>> valid reference json built.'


if __name__ == '__main__':

    # load dataset of the original data format
    load_dataset()

    # observe dataset
    # look_dataset()

    # build_valid_reference_json()

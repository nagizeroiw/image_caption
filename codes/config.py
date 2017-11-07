# contains config of the image caption project.


class Config():

    # ###################### Directory configuration ######################

    # dataset dir
    dataset_dir = '/home/kesu/image_caption/dataset/'
    # result dir
    result_dir = '/home/kesu/image_caption/result/'
    # plots dir
    plot_dir = result_dir + 'plots/'
    # checkpoint dir
    ckpt_dir = result_dir + 'checkpoint/'
    # inferenced json dir
    json_dir = result_dir + 'output/'
    # eval result dir
    eval_dir = result_dir + 'output/'

    # ###################### Dataset ######################

    # maximum length of caption sentences
    # 30 may be too long
    maxlen = 20
    # total number of words in the worddict
    # (indicates the maximum possible value of worddict)
    # actually when words that freq. < 5 are elimated, n_words ~= 9000.
    n_words = 8270
    # dim of pre-trained CNN.
    dim_feature = 1536
    # batch size of SGD (both train and valid)
    batch_size = 64
    # num of examples per epoch
    #  this number is not precise.
    #  And this number is not used any more.
    num_examples = 586363

    # ###################### Model ######################

    # dim of LSTM input (embedding) neurons
    dim_embedding = 512
    # dim of LSTM hidden neurons
    #  3518 in yinpeng`s paper
    dim_hidden = 1024
    # layers of LSTM neurons
    num_layers = 1
    # size of beam search
    beam_size = 3
    # if lstm init state trainable
    lstm_init_state = False
    # if attention-applied image feature is fed into LSTM as input
    visual_attention = False
    # dim of attention-applied image feature
    dim_attention = 512

    # model name
    model_name = 'bigger_orthogonal'

    # ###################### Training ######################

    # whether use cuda
    use_cuda = True
    # number of epochs for training
    n_epoch = 40
    # number of updates (early stopping)
    n_updates = 1000000
    # learning_rate
    learning_rate = 2e-4
    # gradient clip
    gradient_clip = 5.0
    # the frequency of log pringing when training
    train_log_freq = 500
    # the frequency of saving losses for visualization
    train_plot_freq = 500
    # if load training checkpoint
    is_reload = False
    # training checkpoint name
    train_ckpt_name = ckpt_dir + 'training_%s.ckpt' % model_name
    # if debugging (only update once)
    is_debug = False
    # the frequency of plotting a temporary graph
    plot_freq = 25000

    # ###################### Validating, Sampling ######################

    # the frequency of validating
    valid_freq = 10
    # the frequency of validation plotting
    valid_plot_freq = 500
    # the frequency of sampling
    sample_freq = 8000
    # sampling counts (better < batch_size)
    sample_count = 2

    # ###################### Inference, Evaluation ######################

    # the name of checkpoint file used when evaluating
    infer_ckpt_name = train_ckpt_name
    # the frequency of inference during training
    check_freq = 25000
    # eval file
    eval_file = eval_dir + 'result_%s.txt' % model_name
    # plot file
    loss_plot_file = plot_dir + 'loss_%s.png' % model_name
    # eval plot file
    eval_plot_file = plot_dir + 'eval_%s.png' % model_name
    # best model checkpoint file
    best_ckpt_file = ckpt_dir + 'best_%s.ckpt' % model_name
    # the frequency of inference output
    inference_freq = 100
    # output json name
    inference_file = json_dir + 'inferenced_%s.json' % model_name

    # the name of checkpoint file used when testing
    test_ckpt_name = ckpt_dir + 'training.ckpt'
    # output test json file to hand in
    test_inference_file = json_dir + 'test_inferenced.json'

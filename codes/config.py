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

    # ###################### Dataset ######################

    # maximum length of caption sentences
    maxlen = 20
    # total number of words in the worddict
    # (indicates the maximum possible value of worddict)
    n_words = 10500
    # dim of pre-trained CNN.
    dim_feature = 1536
    # batch size of SGD (both train and valid)
    batch_size = 64

    # ###################### Model ######################

    # dim of LSTM input (embedding) neurons
    dim_embedding = 512
    # dim of LSTM hidden neurons
    #  3518 in yinpeng`s paper
    dim_hidden = 1024
    # layers of LSTM neurons
    num_layers = 1

    # ###################### Training ######################

    # whether use cuda
    use_cuda = True
    # number of epochs for training
    n_epoch = 10
    # learning_rate
    learning_rate = 1e-4
    # the frequency of log pringing when training
    train_log_freq = 200
    # the frequency of saving losses for visualization
    train_plot_freq = 500
    # if load training checkpoint
    is_reload = False
    # training checkpoint name
    train_ckpt_name = ckpt_dir + 'training.ckpt'
    # if debugging (only update once)
    is_debug = False
    # the frequency of plotting a temporary graph
    plot_freq = 50000

    # ###################### Validating, Sampling ######################

    # the frequency of validating
    valid_freq = 25
    # the frequency of validation plotting
    valid_plot_freq = 500
    # the frequency of sampling
    sample_freq = 8000
    # sampling counts (better < batch_size)
    sample_count = 4
    # size of beam search
    beam_size = 20

    # ###################### Inference ######################

    # the frequency of inference output
    inference_freq = 100
    # output json name
    inference_file = json_dir + 'inferenced.json'

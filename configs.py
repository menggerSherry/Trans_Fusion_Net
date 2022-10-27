def all_config(mode, fuc, model):
    config = {}

    config['batch_size'] = 16
    config['width'] = 512
    config['height'] = 512
    config['input_channels'] = 3
    config['num_classes'] = 1  # Number of classes
    config['image_path'] = 'datasets/' + mode + '/' + fuc + '/images/'
    config['mask_path'] = 'datasets/' + mode + '/' + fuc + '/masks/'
    config['model_path'] = 'save_model/' + fuc + '/'

    if model == 'Transformer':
        config['img_dim'] = 256                 # 输入图像大小dim*dim
        config['out_dim'] = 65536               # 输出的维度
        config['patch_dim'] = 16                # 每个补丁的大小
        config['num_channels'] = 3              # 图像通道数
        config['embedding_dim'] = 768           # 每个补丁embedding后的维度大小
        config['num_heads'] = 12
        config['num_layers'] = 12
        config['hidden_dim'] = 3072
        config['dropout_rate'] = 0.1
        config['attention_dropout_rate'] = 0.0
        config['use_representation'] = False
        config['conv_patch_representation'] = False
        config['positional_encoding_type'] = 'fixed'         # fixed, learned

    return config
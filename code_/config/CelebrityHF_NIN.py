batch_size   = 1
# Train & evaluate an object classifier (for the Celebrity Face High Frequency rotation task ) with convolutional
# layers on the feature maps of the 2nd conv. block of the RotNet model trained above
config = {}
# set the parameters related to the training and testing set


data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'celebrityh'
data_test_opt['split'] = 'test'

#config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 100
#这里feat_extractor的num_classes是4是否合适？是旋转任务，没有问题。
net_opt = {}
net_opt['num_classes'] = 4
net_opt['num_stages']  = 4
net_opt['use_avg_on_conv3'] = False
networks = {}
pretrained_model_3block = "experiments/CIFAR10_RotNet_NIN4blocks/model_net_epoch200"
pretrained_model_4block='C:/Users/10725/Downloads/model_net_epoch200'
#networks['feat_extractor'] = {'def_file': 'architectures/NetworkInNetwork.py', 'pretrained': feat_pretrained_file, 'opt': feat_net_opt,  'optim_params': None}
net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(60, 0.1),(120, 0.02),(160, 0.004),(200, 0.0008)]}
networks['model'] = {'def_file': 'architectures/NetworkInNetwork.py', 'pretrained': pretrained_model_4block, 'opt': net_opt,  'optim_params': net_optim_params}
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'
config['best_metric'] = 'prec1'

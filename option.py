import argparse
#import template

parser = argparse.ArgumentParser(description='SAP-Net (WBRE module)')

parser.add_argument("--seed", type=int, default=123456789)

parser.add_argument('--config', default='config.yaml', type=str,
                    help='config file path (default: config.yaml)')
parser.add_argument("--log_dir", type=str, default="/home/yl/logger_enhance",
                    help="log directory for Tensorboard log output")
parser.add_argument('--batch_size', type=int, default=12,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--exp_id', default='0', type=str,
                    help='exp id (default: 0)')
parser.add_argument('--disable_gpu', action='store_true',
                    help='flag whether to disable GPU')
parser.add_argument("--log_dir_MW", type=str, default="/home/yl/logger_enhance_MW",
                    help="log directory for Tensorboard log output (MWCNN)")
parser.add_argument('--loss', type=str, default='1*MSE',
                    help='loss function configuration of image restoration '
                         '(1: 0.5*MSE+1*MSE+1*MSE+1*MSE 2: 1*MSE)')
parser.add_argument('--database', default='yl360Dataset', type=str,
                    help='database name (default: LIVE)')
parser.add_argument('--batch_size_iqa', type=int, default=8,
                    help='input batch size for training IQA (default: 128)')
parser.add_argument('--lr_iqa', type=float, default=1e-4,
                    help='learning rate (default: 0.001)')

# parser.add_argument('--resume', default=None, type=str,
#                     help='path to latest checkpoint (default: None)')

parser.add_argument("--log_dir_NonMW", type=str, default="/home/yl/logger_enhance_NonMW",
                    help="log directory for Tensorboard log output (NonMW-MWCNN)")
parser.add_argument("--log_dir_NonMW1", type=str, default="/home/yl/logger_enhance_NonMW1",
                    help="log directory for Tensorboard log output (NonMW-MWCNN1)")
parser.add_argument("--log_dir_IQA", type=str, default="/home/yl/logger_IQA1",
                    help="log directory for Tensorboard log output (IQA) patch=9, enhace.epoch=316")
parser.add_argument("--log_dir_IQA1", type=str, default="/home/yl/logger_IQA2",
                    help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444")
parser.add_argument("--log_dir_IQA2", type=str, default="/home/yl/logger_IQA3",
                    help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (The same as iqa1)")
parser.add_argument("--log_dir_IQA3", type=str, default="/home/yl/logger_IQA4",
                    help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: Non_CBAM)")
parser.add_argument("--log_dir_IQA5", type=str, default="/home/yl/logger_IQA5",
                    help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: Non_CBAMRes)")
parser.add_argument("--log_dir_IQA6", type=str, default="/home/yl/logger_IQA6",
                    help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: quality map mul)")
parser.add_argument("--log_dir_IQA7", type=str, default="/home/yl/logger_IQA7",
                    help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: quality map add)")



# parser.add_argument('--multi_gpu', action='store_true',
#                     help='flag whether to use multiple GPUs')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=12,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')

# Model specifications
parser.add_argument('--model', default='MWCNN',
                    help='model name: DenseWTUnet, MWCNN, ResCBAMIQA')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=12,
                    help='do test per every N batches')


parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications

parser.add_argument('--lr_decay', type=int, default=50,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')


# Loss specifications
# parser.add_argument('--loss', type=str, default='1*MSE',
#                     help='loss function configuration of image restoration (1: 0.5*MSE+1*MSE+1*MSE+1*MSE 2: 1*MSE)')

parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
# options for test
parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')

args = parser.parse_args()
#template.set_template(args)

#args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

# import argparse
# #import template
#
# parser = argparse.ArgumentParser(description='EDSR and MDSR')
#
# parser.add_argument('--batch_size', type=int, default=12,
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--batch_size_iqa', type=int, default=8,
#                     help='input batch size for training IQA (default: 128)')
# parser.add_argument('--epochs', type=int, default=1e20,
#                     help='number of epochs to train (default: 500)')
# parser.add_argument('--lr', type=float, default=0.001,
#                     help='learning rate (default: 0.001)')
# parser.add_argument('--lr_iqa', type=float, default=1e-4,
#                     help='learning rate (default: 0.001)')
# parser.add_argument('--weight_decay', type=float, default=0.0,
#                     help='weight decay (default: 0.0)')
# parser.add_argument('--config', default='config.yaml', type=str,
#                     help='config file path (default: config.yaml)')
# parser.add_argument('--exp_id', default='0', type=str,
#                     help='exp id (default: 0)')
# parser.add_argument('--database', default='yl360Dataset', type=str,
#                     help='database name (default: LIVE)')
#
# # parser.add_argument('--resume', default=None, type=str,
# #                     help='path to latest checkpoint (default: None)')
# parser.add_argument("--log_dir", type=str, default="/home/yl/logger_enhance",
#                     help="log directory for Tensorboard log output")
#
# parser.add_argument("--log_dir_MW", type=str, default="/home/yl/logger_enhance_MW",
#                     help="log directory for Tensorboard log output (MWCNN)")
# parser.add_argument("--log_dir_NonMW", type=str, default="/home/yl/logger_enhance_NonMW",
#                     help="log directory for Tensorboard log output (NonMW-MWCNN)")
# parser.add_argument("--log_dir_NonMW1", type=str, default="/home/yl/logger_enhance_NonMW1",
#                     help="log directory for Tensorboard log output (NonMW-MWCNN1)")
# parser.add_argument("--log_dir_IQA", type=str, default="/home/yl/logger_IQA1",
#                     help="log directory for Tensorboard log output (IQA) patch=9, enhace.epoch=316")
# parser.add_argument("--log_dir_IQA1", type=str, default="/home/yl/logger_IQA2",
#                     help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444")
# parser.add_argument("--log_dir_IQA2", type=str, default="/home/yl/logger_IQA3",
#                     help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (The same as iqa1)")
# parser.add_argument("--log_dir_IQA3", type=str, default="/home/yl/logger_IQA4",
#                     help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: Non_CBAM)")
# parser.add_argument("--log_dir_IQA5", type=str, default="/home/yl/logger_IQA5",
#                     help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: Non_CBAMRes)")
# parser.add_argument("--log_dir_IQA6", type=str, default="/home/yl/logger_IQA6",
#                     help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: quality map mul)")
# parser.add_argument("--log_dir_IQA7", type=str, default="/home/yl/logger_IQA7",
#                     help="log directory for Tensorboard log output (IQA) patch=15, enhace.epoch=444 (abl: quality map add)")
#
#
# parser.add_argument('--disable_gpu', action='store_true',
#                     help='flag whether to disable GPU')
# # parser.add_argument('--multi_gpu', action='store_true',
# #                     help='flag whether to use multiple GPUs')
#
#
# parser.add_argument('--debug', action='store_true',
#                     help='Enables debug mode')
# parser.add_argument('--template', default='.',
#                     help='You can set various templates in option.py')
#
# # Hardware specifications
# parser.add_argument('--n_threads', type=int, default=12,
#                     help='number of threads for data loading')
# parser.add_argument('--cpu', action='store_true',
#                     help='use cpu only')
# parser.add_argument('--n_GPUs', type=int, default=1,
#                     help='number of GPUs')
#
#
# # Data specifications
# parser.add_argument('--dir_data', type=str, default='/share/Dataset/',
#                     help='dataset directory')
# parser.add_argument('--dir_demo', type=str, default='../test',
#                     help='demo image directory')
# parser.add_argument('--data_train', type=str, default='DIV2K',
#                     help='train dataset name')
# parser.add_argument('--data_test', type=str, default='Set5',
#                     help='test dataset name')
# parser.add_argument('--benchmark_noise', action='store_true',
#                     help='use noisy benchmark sets')
# parser.add_argument('--n_train', type=int, default=800,
#                     help='number of training set')
# parser.add_argument('--n_val', type=int, default=5,
#                     help='number of validation set')
# parser.add_argument('--offset_val', type=int, default=800,
#                     help='validation index offest')
# parser.add_argument('--ext', type=str, default='sep_reset',
#                     help='dataset file extension')
# parser.add_argument('--scale', default='2',
#                     help='super resolution scale')
# parser.add_argument('--patch_size', type=int, default=384,
#                     help='output patch size')
# parser.add_argument('--rgb_range', type=int, default=1,
#                     help='maximum value of RGB')
# parser.add_argument('--n_colors', type=int, default=3,
#                     help='number of color channels to use')
# parser.add_argument('--noise', type=str, default='.',
#                     help='Gaussian noise std.')
# parser.add_argument('--chop', action='store_true',
#                     help='enable memory-efficient forward')
#
# # Model specifications
# parser.add_argument('--model', default='ResCBAMIQA',
#                     help='model name: DenseWTUnet, MWCNN, ResCBAMIQA')
#
# parser.add_argument('--act', type=str, default='relu',
#                     help='activation function')
# parser.add_argument('--pre_train', type=str, default='.',
#                     help='pre-trained model directory')
# parser.add_argument('--extend', type=str, default='.',
#                     help='pre-trained model directory')
# parser.add_argument('--n_resblocks', type=int, default=20,
#                     help='number of residual blocks')
# parser.add_argument('--n_feats', type=int, default=64,
#                     help='number of feature maps')
# parser.add_argument('--res_scale', type=float, default=1,
#                     help='residual scaling')
# parser.add_argument('--shift_mean', default=True,
#                     help='subtract pixel mean from the input')
# parser.add_argument('--precision', type=str, default='single',
#                     choices=('single', 'half'),
#                     help='FP precision for test (single | half)')
#
# # Training specifications
# parser.add_argument('--reset', action='store_true',
#                     help='reset the training')
# parser.add_argument('--test_every', type=int, default=12,
#                     help='do test per every N batches')
#
#
# parser.add_argument('--split_batch', type=int, default=1,
#                     help='split the batch into smaller chunks')
# parser.add_argument('--self_ensemble', action='store_true',
#                     help='use self-ensemble method for test')
# parser.add_argument('--test_only', action='store_true',
#                     help='set this option to test the model')
# parser.add_argument('--gan_k', type=int, default=1,
#                     help='k value for adversarial loss')
#
# # Optimization specifications
#
# parser.add_argument('--lr_decay', type=int, default=50,
#                     help='learning rate decay per N epochs')
# parser.add_argument('--decay_type', type=str, default='step',
#                     help='learning rate decay type')
# parser.add_argument('--gamma', type=float, default=0.5,
#                     help='learning rate decay factor for step decay')
# parser.add_argument('--optimizer', default='ADAM',
#                     choices=('SGD', 'ADAM', 'RMSprop'),
#                     help='optimizer to use (SGD | ADAM | RMSprop)')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='SGD momentum')
# parser.add_argument('--beta1', type=float, default=0.9,
#                     help='ADAM beta1')
# parser.add_argument('--beta2', type=float, default=0.999,
#                     help='ADAM beta2')
# parser.add_argument('--epsilon', type=float, default=1e-8,
#                     help='ADAM epsilon for numerical stability')
#
#
# # Loss specifications
# parser.add_argument('--loss', type=str, default='1*MSE',
#                     help='loss function configuration of image restoration '
#                          '(1: 0.5*MSE+1*MSE+1*MSE+1*MSE 2: 1*MSE)')
#
# parser.add_argument('--skip_threshold', type=float, default='1e6',
#                     help='skipping batch that has large error')
#
# # Log specifications
# parser.add_argument('--save', type=str, default='test',
#                     help='file name to save')
# parser.add_argument('--load', type=str, default='.',
#                     help='file name to load')
# parser.add_argument('--resume', type=int, default=0,
#                     help='resume from specific checkpoint')
# parser.add_argument('--print_model', action='store_true',
#                     help='print model')
# parser.add_argument('--save_models', action='store_true',
#                     help='save all intermediate models')
# parser.add_argument('--print_every', type=int, default=100,
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--save_results', action='store_true',
#                     help='save output results')
#
# # options for residual group and feature channel reduction
# parser.add_argument('--n_resgroups', type=int, default=10,
#                     help='number of residual groups')
# parser.add_argument('--reduction', type=int, default=16,
#                     help='number of feature maps reduction')
# # options for test
# parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
#                     help='dataset directory for testing')
# parser.add_argument('--testset', type=str, default='Set5',
#                     help='dataset name for testing')
#
# args = parser.parse_args()
# #template.set_template(args)
#
# args.scale = list(map(lambda x: int(x), args.scale.split('+')))
#
# if args.epochs == 0:
#     args.epochs = 1e8
#
# for arg in vars(args):
#     if vars(args)[arg] == 'True':
#         vars(args)[arg] = True
#     elif vars(args)[arg] == 'False':
#         vars(args)[arg] = False

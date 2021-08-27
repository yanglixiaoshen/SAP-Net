import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale  # 超分辨路重建的 scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision  # 预测结果的浮点数精度
        self.cpu = args.cpu  # 如果传入cpu的外部参数，手动声明使用cpu，就用cpu，否则用gpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs  # gpu的个数，默认1个
        self.save_models = args.save_models  # 是否选择保存模型

        # 导入需要使用的预测模型
        module = import_module('model.' + args.model)
        self.model = module.make_model(args).to(self.device)

        # 浮点格式的精度
        if args.precision == 'half': self.model.half()

        # 多GPU
        # if not args.cpu and args.n_GPUs > 1:
        #     self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # 导入外部的预训练模型
        # self.load(ckp.dir,
        #           pre_train=args.pre_train,
        #           resume=args.resume,
        #           name=args.model,
        #           cpu=args.cpu)

        # 输出模型
        if args.print_model:
            print(self.model)

    def forward(self, x):
        return self.model(x)

    # 根据 GPU 个数导入训练模型
    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    # 对模型进行参数映射
    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    # 保存模型
    def save(self, apath, epoch, name, is_best=False):
        target = self.get_model()
        torch.save(target.state_dict(),
                   os.path.join(apath, 'model', name + 'model_latest.pt'))
        if is_best:
            torch.save(target.state_dict(),
                       os.path.join(apath, 'model', name + 'model_best.pt'))
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model',
                             name + 'model_{}.pt'.format(epoch)))

    # 导入模型
    def load(self, apath, pre_train='.', resume=-1, name='', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(torch.load(
                os.path.join(pre_train, name + 'model_latest.pt'), **kwargs),
                                             strict=False)

            # self.get_model().load_state_dict(
            #     torch.load(
            #         os.path.join(apath, 'model', name + 'model_latest.pt'),
            #         **kwargs
            #     ),
            #     strict=False
            # )

        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(torch.load(
                    pre_train, **kwargs),
                                                 strict=False)
        else:
            self.get_model().load_state_dict(torch.load(
                os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                **kwargs),
                                             strict=False)
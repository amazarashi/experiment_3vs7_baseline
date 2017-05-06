import argparse
from chainer import optimizers
import darknet19
import amaz_imagenet
import amaz_augumentationCustom
import amaz_optimizer
import amaz_trainer_batchInbatch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--epoch', '-e', type=int,
                        default=60,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=64,
                        help='mini batch number')
    parser.add_argument('--batchinbatch', '-bb', type=int,
                        default=16,
                        help='batch in batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu, put gpu id here')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='learning rate')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')
    epoch = args.pop('epoch')

    imagenet = amaz_imagenet.ImageNet()
    dataset = imagenet.loader()
    model = darknet19.Darknet19(category_num=1000)
    optimizer = amaz_optimizer.OptimizerDarknet(model,lr=0.04,epoch=60,batch=args.pop("batch"))
    dataaugumentation = amaz_augumentationCustom.Normalize224
    args['model'] = model
    args['optimizer'] = optimizer
    args['dataset'] = dataset
    args['dataaugumentation'] = dataaugumentation
    main = amaz_trainer_batchInbatch.Trainer(**args)
    main.run()

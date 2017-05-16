import argparse
from chainer import optimizers
import darknet19
import amaz_cifar10_dl
import amaz_augumentationCustom
import amaz_optimizer
import amaz_trainer_batchInbatch_kmeans

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--epoch', '-e', type=int,
                        default=60,
                        help='maximum epoch')
    parser.add_argument('--batchinbatch', '-bb', type=int,
                        default=16,
                        help='batch in batch number')
    parser.add_argument('--batch', '-b', type=int,
                        default=64,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu, put gpu id here')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='learning rate')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')
    epoch = args.pop('epoch')

    dataset = amaz_cifar10_dl.Cifar10().categorical_loader()
    #dataset = amaz_cifar10_dl.Cifar10().loader()
    elseIndices = [0,1,2]
    category_num = 10 - len(elseIndices)
    model = darknet19.Darknet19(category_num=category_num)
    optimizer = amaz_optimizer.OptimizerDarknet(model,lr=0.04,epoch=300,batch=args.pop("batch"))
    dataaugumentation = amaz_augumentationCustom.Normalize224
    args['model'] = model
    args['optimizer'] = optimizer
    args['dataset'] = dataset
    args['dataaugumentation'] = dataaugumentation
    args['elseIndices'] = elseIndices
    main = amaz_trainer_batchInbatch_kmeans.Trainer(**args)
    main.run()

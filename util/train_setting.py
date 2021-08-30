__all__ = ['scaling', 'loss_fn', 'optim']

from .headers import *


def scaling(data, method='minmax'):
    if method == 'minmax':
        scale = MinMaxScaler()

    elif method == 'standard':
        scale = StandardScaler()

    elif method == 'robust':
        scale = RobustScaler()

    else:
        raise f'Not Support {method} scale'

    scale_data = scale.fit_transform(data)

    return scale, scale_data


def loss_fn(args):
    if args.loss_fn == 'huber':
        loss = Huber()
    elif args.loss_fn == 'mse':
        loss = mse()
    elif args.loss_fn == 'mae':
        loss = mae()
    elif args.loss_fn == 'binary':
        loss = binary_crossentropy()
    elif args.loss_fn == 'category':
        loss = categorical_crossentropy()
    else:
        raise f'{args.loss_fn} == not support'

    return loss


def optim(args):
    if args.optim == 'adam':
        optimizer = Adam(learning_rate=args.lr)
    elif args.optim == 'sgd':
        optimizer = SGD(learning_rate=args.lr)
    elif args.optim == "rmsprop":
        optimizer = RMSprop(learning_rate=args.lr)
    else:
        raise f'{args.optim} == not support'

    return optimizer

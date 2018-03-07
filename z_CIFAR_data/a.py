def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


temp = unpickle('cifar10batchespy/data_batch_1')

print(type(temp))

print(temp)
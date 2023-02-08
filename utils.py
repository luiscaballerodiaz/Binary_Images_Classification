import os
import shutil
import random


def create_working_folders(base_dir, folds, test_size):
    """Split the input images in train, validation and test sets and store the images in the corresponding folders"""
    base_cars_dir = os.path.join(base_dir, 'Cars')
    base_bikes_dir = os.path.join(base_dir, 'Bikes')
    ncars = os.listdir(base_cars_dir)
    nbikes = os.listdir(base_bikes_dir)
    random.Random(0).shuffle(ncars)
    random.Random(0).shuffle(nbikes)
    print('Number of images in Cars folder: {}'.format(len(ncars)))
    print('Number of images in Bikes folder: {}\n'.format(len(nbikes)))
    train_dir, test_dir = train_test_split(nbikes, ncars, test_size, base_dir, base_bikes_dir, base_cars_dir)
    trainval_dir, testval_dir, trainval_count, testval_count = cross_validation_split(folds, base_dir, base_cars_dir,
                                                                                      base_bikes_dir, test_size)
    return train_dir, test_dir, trainval_dir, testval_dir, trainval_count, testval_count


def train_test_split(nbikes, ncars, test_size, base_dir, base_bikes_dir, base_cars_dir):
    test_dir = os.path.join(base_dir, 'Test')
    test_cars_dir = os.path.join(test_dir, 'Cars')
    test_bikes_dir = os.path.join(test_dir, 'Bikes')
    train_dir = os.path.join(base_dir, 'Train')
    train_cars_dir = os.path.join(train_dir, 'Cars')
    train_bikes_dir = os.path.join(train_dir, 'Bikes')
    if os.path.exists(os.path.join(base_dir, 'Train')) is False:
        os.mkdir(test_dir)
        os.mkdir(test_cars_dir)
        os.mkdir(test_bikes_dir)
        os.mkdir(train_dir)
        os.mkdir(train_cars_dir)
        os.mkdir(train_bikes_dir)
        for fname, i in zip(nbikes, range(len(nbikes))):
            if i < round(len(nbikes) * (1 - test_size)):
                dst = os.path.join(train_bikes_dir, fname)
            else:
                dst = os.path.join(test_bikes_dir, fname)
            src = os.path.join(base_bikes_dir, fname)
            shutil.copyfile(src, dst)
        for fname, i in zip(ncars, range(len(ncars))):
            if i < round(len(ncars) * (1 - test_size)):
                dst = os.path.join(train_cars_dir, fname)
            else:
                dst = os.path.join(test_cars_dir, fname)
            src = os.path.join(base_cars_dir, fname)
            shutil.copyfile(src, dst)
    print('Total training CARS images: {}'.format(len(os.listdir(train_cars_dir))))
    print('Total training BIKES images: {}'.format(len(os.listdir(train_bikes_dir))))
    print('Total test CARS images: {}'.format(len(os.listdir(test_cars_dir))))
    print('Total test BIKES images: {}\n'.format(len(os.listdir(test_bikes_dir))))
    return train_dir, test_dir


def cross_validation_split(folds, base_dir, base_cars_dir, base_bikes_dir, test_size):
    trainval_dir = [''] * folds
    testval_dir = [''] * folds
    for i in range(folds):
        trainval_dir[i] = os.path.join(base_dir, 'Train' + str(i + 1))
        trainval_cars_dir = os.path.join(trainval_dir[i], 'Cars')
        trainval_bikes_dir = os.path.join(trainval_dir[i], 'Bikes')
        testval_dir[i] = os.path.join(base_dir, 'Validation' + str(i + 1))
        testval_cars_dir = os.path.join(testval_dir[i], 'Cars')
        testval_bikes_dir = os.path.join(testval_dir[i], 'Bikes')
        if os.path.exists(os.path.join(base_dir, 'Train' + str(i + 1))) is False:
            ncars = os.listdir(base_cars_dir)
            random.Random(i + 1).shuffle(ncars)
            nbikes = os.listdir(base_bikes_dir)
            random.Random(i + 1).shuffle(nbikes)
            os.mkdir(trainval_dir[i])
            os.mkdir(trainval_cars_dir)
            os.mkdir(trainval_bikes_dir)
            os.mkdir(testval_dir[i])
            os.mkdir(testval_cars_dir)
            os.mkdir(testval_bikes_dir)
            for fname, j in zip(nbikes, range(round(len(nbikes) * (1 - test_size)))):
                if j < round(len(nbikes) * (1 - test_size) * (folds - 1) / folds):
                    dst = os.path.join(trainval_bikes_dir, fname)
                else:
                    dst = os.path.join(testval_bikes_dir, fname)
                src = os.path.join(base_bikes_dir, fname)
                shutil.copyfile(src, dst)
            for fname, j in zip(ncars, range(round(len(ncars) * (1 - test_size)))):
                if j < round(len(ncars) * (1 - test_size) * (folds - 1) / folds):
                    dst = os.path.join(trainval_cars_dir, fname)
                else:
                    dst = os.path.join(testval_cars_dir, fname)
                src = os.path.join(base_cars_dir, fname)
                shutil.copyfile(src, dst)
        print('Total trainval CARS images fold {}: {}'.format(i + 1, len(os.listdir(trainval_cars_dir))))
        print('Total training BIKES images fold {}: {}'.format(i + 1, len(os.listdir(trainval_bikes_dir))))
        print('Total test CARS images fold {}: {}'.format(i + 1, len(os.listdir(testval_cars_dir))))
        print('Total test BIKES images fold {}: {}\n'.format(i + 1, len(os.listdir(testval_bikes_dir))))
    trainval_count = len(os.listdir(trainval_cars_dir)) + len(os.listdir(trainval_bikes_dir))
    testval_count = len(os.listdir(testval_cars_dir)) + len(os.listdir(testval_bikes_dir))
    return trainval_dir, testval_dir, trainval_count, testval_count

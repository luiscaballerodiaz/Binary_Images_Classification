import os
import shutil
import random


class DataSplit:
    def __init__(self, base_dir, folds, test_size):
        self.base_dir = base_dir
        self.folds = folds
        self.split_size = 1 / folds
        self.test_size = test_size
        self.base_cars_dir = os.path.join(base_dir, 'Cars')
        self.base_bikes_dir = os.path.join(base_dir, 'Bikes')
        self.ncars = os.listdir(self.base_cars_dir)
        self.nbikes = os.listdir(self.base_bikes_dir)
        self.test_dir = os.path.join(base_dir, 'Test')
        self.test_cars_dir = os.path.join(self.test_dir, 'Cars')
        self.test_bikes_dir = os.path.join(self.test_dir, 'Bikes')
        self.train_dir = os.path.join(base_dir, 'Train')
        self.train_cars_dir = os.path.join(self.train_dir, 'Cars')
        self.train_bikes_dir = os.path.join(self.train_dir, 'Bikes')
        self.trainval_dir = [''] * self.folds
        self.testval_dir = [''] * self.folds
        self.train_ncars = None
        self.train_nbikes = None
        self.train_count = 0
        self.test_count = 0
        self.trainval_count = 0
        self.testval_count = 0

    def create_working_folders(self):
        """Split the input images in train, validation and test sets and store images in the corresponding folders"""
        random.Random(0).shuffle(self.ncars)
        random.Random(0).shuffle(self.nbikes)
        print('Number of images in Cars folder: {}'.format(len(self.ncars)))
        print('Number of images in Bikes folder: {}\n'.format(len(self.nbikes)))
        self.train_test_split()
        self.cross_validation_split()
        train_dirs = [self.train_dir, self.train_cars_dir, self.train_bikes_dir, self.trainval_dir]
        test_dirs = [self.test_dir, self.test_cars_dir, self.test_bikes_dir, self.testval_dir]
        counts = [self.trainval_count, self.testval_count, self.train_count, self.test_count]
        return train_dirs, test_dirs, counts

    def train_test_split(self):
        if os.path.exists(os.path.join(self.base_dir, 'Train')) is False:
            os.mkdir(self.test_dir)
            os.mkdir(self.test_cars_dir)
            os.mkdir(self.test_bikes_dir)
            os.mkdir(self.train_dir)
            os.mkdir(self.train_cars_dir)
            os.mkdir(self.train_bikes_dir)
            for fname, i in zip(self.nbikes, range(len(self.nbikes))):
                if i < round(len(self.nbikes) * (1 - self.test_size)):
                    dst = os.path.join(self.train_bikes_dir, fname)
                else:
                    dst = os.path.join(self.test_bikes_dir, fname)
                src = os.path.join(self.base_bikes_dir, fname)
                shutil.copyfile(src, dst)
            for fname, i in zip(self.ncars, range(len(self.ncars))):
                if i < round(len(self.ncars) * (1 - self.test_size)):
                    dst = os.path.join(self.train_cars_dir, fname)
                else:
                    dst = os.path.join(self.test_cars_dir, fname)
                src = os.path.join(self.base_cars_dir, fname)
                shutil.copyfile(src, dst)
        print('Total training CARS images: {}'.format(len(os.listdir(self.train_cars_dir))))
        print('Total training BIKES images: {}'.format(len(os.listdir(self.train_bikes_dir))))
        print('Total test CARS images: {}'.format(len(os.listdir(self.test_cars_dir))))
        print('Total test BIKES images: {}\n'.format(len(os.listdir(self.test_bikes_dir))))
        self.train_ncars = os.listdir(self.train_cars_dir)
        self.train_nbikes = os.listdir(self.train_bikes_dir)
        self.train_count = len(os.listdir(self.train_cars_dir)) + len(os.listdir(self.train_bikes_dir))
        self.test_count = len(os.listdir(self.test_cars_dir)) + len(os.listdir(self.test_bikes_dir))

    def cross_validation_split(self):
        for i in range(self.folds):
            self.trainval_dir[i] = os.path.join(self.base_dir, 'Train' + str(i + 1))
            trainval_cars_dir = os.path.join(self.trainval_dir[i], 'Cars')
            trainval_bikes_dir = os.path.join(self.trainval_dir[i], 'Bikes')
            self.testval_dir[i] = os.path.join(self.base_dir, 'Validation' + str(i + 1))
            testval_cars_dir = os.path.join(self.testval_dir[i], 'Cars')
            testval_bikes_dir = os.path.join(self.testval_dir[i], 'Bikes')
            if os.path.exists(os.path.join(self.base_dir, 'Train' + str(i + 1))) is False:
                os.mkdir(self.trainval_dir[i])
                os.mkdir(trainval_cars_dir)
                os.mkdir(trainval_bikes_dir)
                os.mkdir(self.testval_dir[i])
                os.mkdir(testval_cars_dir)
                os.mkdir(testval_bikes_dir)
                for fname, j in zip(self.train_nbikes, range(round(len(self.train_nbikes)))):
                    if (i * round(len(self.train_nbikes) * self.split_size)) <= j < \
                            ((i + 1) * round(len(self.train_nbikes) * self.split_size)):
                        dst = os.path.join(testval_bikes_dir, fname)
                    else:
                        dst = os.path.join(trainval_bikes_dir, fname)
                    src = os.path.join(self.base_bikes_dir, fname)
                    shutil.copyfile(src, dst)
                for fname, j in zip(self.train_ncars, range(round(len(self.train_ncars)))):
                    if (i * round(len(self.train_ncars) * self.split_size)) <= j < \
                            ((i + 1) * round(len(self.train_ncars) * self.split_size)):
                        dst = os.path.join(testval_cars_dir, fname)
                    else:
                        dst = os.path.join(trainval_cars_dir, fname)
                    src = os.path.join(self.base_cars_dir, fname)
                    shutil.copyfile(src, dst)
            print('Total trainval CARS images fold {}: {}'.format(i + 1, len(os.listdir(trainval_cars_dir))))
            print('Total trainval BIKES images fold {}: {}'.format(i + 1, len(os.listdir(trainval_bikes_dir))))
            print('Total testval CARS images fold {}: {}'.format(i + 1, len(os.listdir(testval_cars_dir))))
            print('Total testval BIKES images fold {}: {}\n'.format(i + 1, len(os.listdir(testval_bikes_dir))))
            if i == 0:
                self.trainval_count = len(os.listdir(trainval_cars_dir)) + len(os.listdir(trainval_bikes_dir))
                self.testval_count = len(os.listdir(testval_cars_dir)) + len(os.listdir(testval_bikes_dir))

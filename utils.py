import os
import shutil


def create_working_folders(base_dir, validation_size, test_size):
    train_dir = os.path.join(base_dir, 'Train')
    train_cars_dir = os.path.join(train_dir, 'Cars')
    train_bikes_dir = os.path.join(train_dir, 'Bikes')

    test_dir = os.path.join(base_dir, 'Test')
    test_cars_dir = os.path.join(test_dir, 'Cars')
    test_bikes_dir = os.path.join(test_dir, 'Bikes')

    validation_dir = os.path.join(base_dir, 'Validation')
    validation_cars_dir = os.path.join(validation_dir, 'Cars')
    validation_bikes_dir = os.path.join(validation_dir, 'Bikes')

    if os.path.exists(os.path.join(base_dir, 'Train')) is False:
        base_cars_dir = os.path.join(base_dir, 'Cars')
        ncars = os.listdir(base_cars_dir)
        print('Number of images in Cars folder: {}'.format(len(ncars)))
        base_bikes_dir = os.path.join(base_dir, 'Bikes')
        nbikes = os.listdir(base_bikes_dir)
        print('Number of images in Bikes folder: {}\n'.format(len(nbikes)))

        os.mkdir(train_dir)
        os.mkdir(train_cars_dir)
        os.mkdir(train_bikes_dir)

        os.mkdir(test_dir)
        os.mkdir(test_cars_dir)
        os.mkdir(test_bikes_dir)

        os.mkdir(validation_dir)
        os.mkdir(validation_cars_dir)
        os.mkdir(validation_bikes_dir)

        test_cars_images = round(len(ncars) * test_size)
        test_bikes_images = round(len(nbikes) * test_size)
        validation_cars_images = round(len(ncars) * validation_size)
        validation_bikes_images = round(len(nbikes) * validation_size)
        train_cars_images = len(ncars) - validation_cars_images - test_cars_images
        train_bikes_images = len(nbikes) - validation_bikes_images - test_bikes_images

        for fname, i in zip(nbikes, range(len(nbikes))):
            if i < train_bikes_images:
                dst = os.path.join(train_bikes_dir, fname)
            elif i < (train_bikes_images + validation_bikes_images):
                dst = os.path.join(validation_bikes_dir, fname)
            else:
                dst = os.path.join(test_bikes_dir, fname)
            src = os.path.join(base_bikes_dir, fname)
            shutil.copyfile(src, dst)

        for fname, i in zip(ncars, range(len(ncars))):
            if i < train_cars_images:
                dst = os.path.join(train_cars_dir, fname)
            elif i < (train_cars_images + validation_cars_images):
                dst = os.path.join(validation_cars_dir, fname)
            else:
                dst = os.path.join(test_cars_dir, fname)
            src = os.path.join(base_cars_dir, fname)
            shutil.copyfile(src, dst)

    print('Total training CARS images:', len(os.listdir(train_cars_dir)))
    print('Total training BIKES images:', len(os.listdir(train_bikes_dir)))
    print('Total validation CARS images:', len(os.listdir(validation_cars_dir)))
    print('Total validation BIKES images:', len(os.listdir(validation_bikes_dir)))
    print('Total test CARS images:', len(os.listdir(test_cars_dir)))
    print('Total test BIKES images:', len(os.listdir(test_bikes_dir)))
    print('\n')
    return train_dir, test_dir, validation_dir

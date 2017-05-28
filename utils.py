def load_imgs(datapath, classes, target_size):
    import glob
    import numpy as np
    from PIL import Image
    from keras.preprocessing import image
    import os

    x = []
    y = []

    for cls in classes:
        # traverse over folders
        for f in glob.glob(os.path.join(datapath, cls, '*.jpg')):
            img = Image.open(str(f))
            img = img.convert('RGB')
            img = img.resize(target_size)
            arr = image.img_to_array(img)
            x.append(arr)
            y.append(cls)
        # for some files with capital extension letter
        for f in glob.glob(os.path.join(datapath, cls, '*.JPG')):
            img = Image.open(str(f))
            img = img.convert('RGB')
            img = img.resize(target_size)
            arr = image.img_to_array(img)
            x.append(arr)
            y.append(cls)

    x = np.array(x)
    y = np.array(y)

    return x, y
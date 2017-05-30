def load_imgs(datapath, classes, target_size):
    import glob
    import numpy as np
    from PIL import Image
    from keras.preprocessing import image
    from sklearn.feature_extraction.image import extract_patches_2d
    import os

    x = []
    y = []

    for cls in classes:
        # traverse over folders
        for f in glob.glob(os.path.join(datapath, cls, '*.jpg')):
            img = Image.open(str(f))
            img = img.convert('RGB')
            img = img.resize((target_size[0], target_size[1]))
            arr = image.img_to_array(img)
            # crop the images
            arr_ = extract_patches_2d(arr, patch_size=target_size)
            for crop in arr_:
                x.append(crop)
                y.append(cls)
        # for some files with capital extension letter
        for f in glob.glob(os.path.join(datapath, cls, '*.JPG')):
            img = Image.open(str(f))
            img = img.convert('RGB')
            img = img.resize((target_size[0], target_size[1]))
            arr = image.img_to_array(img)
            # crop the images
            arr_ = extract_patches_2d(arr, patch_size=target_size)
            for crop in arr_:
                x.append(crop)
                y.append(cls)

    x = np.array(x)
    y = np.array(y)

    return x, y
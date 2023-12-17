import os.path as osp
import os
from torchvision.datasets import PCAM
import errno

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def extract_and_save_image(dataset, save_dir, categories):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        # class_dir = osp.join(save_dir, str(label).zfill(3))
        class_dir = osp.join(save_dir, categories[label])
        mkdir_if_missing(class_dir)
        impath = osp.join(class_dir, str(i + 1).zfill(5) + ".jpg")
        img.save(impath)

def download_and_prepare(name, root):
    print("Dataset: {}".format(name))
    print("Root: {}".format(root))
    
    if name == "pcam":
        #download manually if there is an error : https://github.com/basveeling/pcam
        train = PCAM(root, split="train", download=False)
        test = PCAM(root, split="test", download=False)
        val = PCAM(root, split="val", download=False)
    else:
        raise ValueError
    
    train_dir = osp.join(root, name, "train")
    test_dir = osp.join(root, name, "test")
    val_dir = osp.join(root, name, "val")

    if name == 'pcam':
        categories = ['negative', 'positive']
    else:
        categories = train.classes

    extract_and_save_image(train, train_dir, categories)
    extract_and_save_image(test, test_dir, categories)
    extract_and_save_image(val,val_dir, categories)
if __name__ == "__main__":
    download_and_prepare("pcam", 'ROOT')
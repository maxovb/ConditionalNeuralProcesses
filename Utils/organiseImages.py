import shutil
import os
from pathlib import Path

def main():
    cwd_path = Path(os.getcwd())
    parent = cwd_path.parent

    source = os.path.join(parent, "img_align_celeba")
    train_path = os.path.join(parent, "DataSets", "CelebA", "train")
    test_path = os.path.join(parent, "DataSets", "CelebA", "test")

    files = os.listdir(source)

    for f in files:
        if int(f.split(".")[0]) <= 182400:
            shutil.move(os.path.join(source, f), train_path)
        else:
            shutil.move(os.path.join(source, f), test_path)


if __name__ == '__main__':
    main()
import hashlib
import os
import os.path


def md5sum(filename):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == "__main__":
    folder = "images/Lennart/"
    dupe_folder = folder + "duplicates/"
    thumbs_folder = folder + "thumbs/"
    thumbs_dupe_folder = thumbs_folder + "duplicates/"
    if not os.path.exists(dupe_folder):
        os.mkdir(dupe_folder)
    if not os.path.exists(thumbs_dupe_folder):
        os.mkdir(thumbs_dupe_folder)
    print(os.listdir(folder))
    md5sums = dict()
    with os.scandir(folder) as it:
        for file in it:
            if not file.is_file():
                continue
            md5 = md5sum(folder + file.name)
            if md5 in md5sums:
                print(file.name + " is a duplicate of " + md5sums[md5])
                thumb_name = 'th' + file.name
                os.rename(folder + file.name, dupe_folder + file.name)
                os.rename(thumbs_folder + thumb_name, thumbs_dupe_folder + thumb_name)
            else:
                md5sums[md5] = file.name

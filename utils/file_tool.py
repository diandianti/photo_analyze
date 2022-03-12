#!/usr/bin/env python3

import pickle
import hashlib
import os

import cv2
import PIL.Image as Im
import numpy as np
import collections

class FileTool():
    pass

def img_open_cv(img_name):
    # print("cv2 open")
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    return img

def img_open_pil(img_name):
    img = Im.open(img_name)
    img = np.array(img)
    return img

class CacheTool(FileTool):

    def __init__(self, name: str, tmp_dir=None):
        self.name = name
        self.hash_str = hashlib.sha256(name.encode('utf-8')).hexdigest()[:20]

        if type(tmp_dir) == str:
            self.dir = tmp_dir
        else:
            self.dir = ".cache"

        self.file_name = os.path.join(self.dir, self.hash_str) + ".cache"

        if not os.path.isdir(self.dir):
            try:
                os.makedirs(self.dir)
            except:
                raise IOError("Make cache dir fail!") 


    def load(self):

        if not os.path.isfile(self.file_name):
            return None

        with open(self.file_name, "rb") as f:
            data = pickle.load(f)
            return data

    def save(self, data, ow=False):

        if os.path.isfile(self.file_name) and not ow:
            return

        with open(self.file_name, "wb+") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# class CacheFile(CacheTool):
#
#     def __init__(self, name: str, tmp_dir=None):
#         super.__init__(name, tmp_dir)
#
#
#
#     def get(self):
#
#         tmp = self.load()
#         if tmp:
#             return tmp
#         else:
#             return get_method()

found_res = collections.namedtuple("FOUND_RES", ["id", "embs", "imgs"])
class Res():
    def __init__(self, outname: str, dst: str):
        self.outname = outname
        self.all_found = {}
        self.dst = dst

    def save(self, name=None):
        w_name = self.outname
        if name:
            w_name = name
        try:
            with open(w_name, "w+") as f:
                for id, v in self.all_found.items():
                    for img in v.imgs:
                        f.write("%d, %s\n" % (v.id, img))

                    f.write("\n")
        except Exception as e:
            print("Save res file fail!")
            print(e)

    def copy_images(self, dst=None):
        _d = self.dst
        if dst:
            _d = dst

        import shutil
        for id, v in self.all_found.items():

            tar_dir = os.path.join(dst, str(v.id))
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)

            try:
                for img in v.imgs:
                    shutil.copy2(img, tar_dir)
            except:
                print("Copy file fail!")
                continue


def CLI_test():

    import numpy as np
    data = []
    data.append({"name":"value"})
    data.append([1,2,3,4])
    data.append(np.array([1,2,3,4]))
    

    for idx,i in enumerate(data):
        cache_f = CacheTool(str(idx))
        cache_f.save(i)
        get = cache_f.load()

        print(type(i))
        print(i)
        print(type(get))
        print(get)

if __name__ == "__main__":
    import fire
    fire.Fire(CLI_test)

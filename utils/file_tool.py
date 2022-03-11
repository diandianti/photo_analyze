#!/usr/bin/env python3

import pickle
import hashlib
import os

class FileTool():
    pass

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

#!/usr/bin/env python3

import os,sys
from tqdm import tqdm

img_type = ["jpg", "jpeg", "png", "bmp"]

if sys.platform.find("linux") == 0:
    div_char = "/"
else:
    div_char = "\\"

def get_all(top_path: str, ftype = img_type, res_type=None) -> list:
    
    all = []
    all_dict = []

    _w = os.walk(top_path)
    target = 0

    for p,ds,fs in tqdm(_w):

        if len(fs) == 0:
                continue

        _fs = []

        for one_f in fs:

            ex_name = one_f.split(".")[-1]
            f_name = one_f.split(div_char)[-1]

            if ex_name.lower() in ftype and f_name[0] != ".":
                all.append(os.path.join(p, one_f))
                _fs.append(one_f)

        if len(_fs) == 0:
            continue

        all_dict.append({
            "path": p,
            "files": _fs,
            "target": target
            })
        target += 1

    return all if not res_type else all_dict


if __name__ == "__main__":
    print(get_all("./"))

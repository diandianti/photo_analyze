#!/usr/bin/env python
# coding: utf-8

import collections
import numpy as np
import PIL.Image as Im

import utils.file_tool as f_tool
import utils.path_tool as p_tool
from nn.fd_with_lm import Fdlm
import nn.nn_run as nn_run
import cv2
import os
import utils.config as cfg

from nn.align_faces import warp_and_crop_face, get_reference_facial_points

from tqdm import tqdm

found_res = collections.namedtuple("FOUND_RES", ["id", "embs", "imgs"])
temp_res = collections.namedtuple("TEMP_RES", ["id", "dis"])

def align_face(img, lm: np.ndarray, output_size):
    # _, facial5points = detector.detect_faces(img)
    # facial5points = np.reshape(facial5points[0], (2, 5))
    lm = lm.transpose((1, 0))
    lm = lm.flatten()
    lm = lm.reshape((2, 5))

    default_square = True
    inner_padding_factor = 0.05
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)
    dst_img = warp_and_crop_face(img, lm, reference_pts=reference_5pts, crop_size=output_size)

    cv2.imwrite('align.jpg', dst_img)

    return dst_img

def eucd_dis(a, b):
    return np.sqrt( ( (a - b) ** 2 ).sum() )

def copy_images(all_found, dst):

    if not dst:
        print("No dst path!")
        return

    print("Copy img to dst!")
    import shutil
    for id, v in all_found.items():

        tar_dir = os.path.join(dst, str(v.id))
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)

        try:
            for img in v.imgs:
                shutil.copy2(img, tar_dir)
        except:
            print("Copy fail!")
            continue

def _cluster(img_dir: str, fd_model, fr_model, outname, do_copy, dst):

    # get all img list
    all_img = p_tool.get_all(img_dir)

    # init nn
    nn_fd = Fdlm(fd_model,
                 confidence_threshold=cfg.fd.get("confidence_threshold"),
                 top_k=cfg.fd.get("top_k"),
                 keep_top_k=cfg.fd.get("keep_top_k"),
                 nms_threshold=cfg.fd.get("nms_threshold"))
    nn_fr = nn_run.Fr(fr_model)

    fr_input_size = nn_fr.input_size

    all_found = {}
    idx = 0
    v_thd = cfg.fd.get("v_thd", 0.7)
    dis_thd = cfg.cluster.get("dis_thd", 0.8)

    for img in tqdm(all_img):
        _img = Im.open(img)
        imga = np.array(_img)
        bboxes, confs, lms = nn_fd.run(imga)

        embs = []
        for b,c,l in zip(bboxes, confs, lms):
            if c < v_thd:
                continue
            img_crop = align_face(imga, l, fr_input_size)
            emb = nn_fr.run(img_crop)
            embs.append(emb)

        for emb in embs:
            temp = []
            for id,v in all_found.items():
                for e in v.embs:
                    dis = eucd_dis(emb, e)
                    if dis < dis_thd:
                        temp.append(temp_res(id, dis))

            if temp:
                temp.sort(key=lambda x: x.dis)
                target_id = temp[0].id
                all_found[target_id].embs.append(emb)
                all_found[target_id].imgs.append(img)
            else:
                idx += 1
                target_id = idx
                all_found[target_id] = found_res(target_id, [emb], [img])

    with open(outname, "w+") as f:
        for id,v in all_found.items():
            for img in v.imgs:
                f.write("%d, %s\n"%(v.id, img))

            f.write("\n")

    if do_copy:
        copy_images(all_found, dst)


def get_reg_emb(all_reg, nn_fr):

    res_emb = []

    for p in tqdm(all_reg):
        id = p["path"].split("/")[-1]
        embs = []
        for f in p["files"]:
            img_p = p["path"] + "/" + f
            emb = nn_fr.run(img_p)
            embs.append(emb)

        res_emb.append({
            "id": id,
            "embs": embs
            })

    return res_emb

def _compare(pp, reg_path):

    all_reg = p_tool.get_all(reg_path, res_type="dict")
    pro_path = pp
    all_pro = p_tool.get_all(pro_path)

    fr_path = "/datasets/mask_face/export/maskfr/less.onnx"
    nn_fr = nn_run.Fr(fr_path)

    mask_embf = f_tool.CacheTool("mask")

    tmp = mask_embf.load()
    if tmp:
        res_emb = tmp
    else:
        res_emb = get_reg_emb(all_reg, nn_fr)
        mask_embf.save(res_emb)


    dis_thd = 0.75
    for one in tqdm(all_pro):
        sames = []

        one_emb = nn_fr.run(one)
        for r in res_emb:
            for e in r["embs"]:
                dis = eucd_dis(e, one_emb)
                if dis < dis_thd:
                    sames.append(temp_res(r["id"], dis))

        if len(sames):
            continue

        sames.sort(key=lambda x: x.dis)
        sames.append()

class CLI():
    def cluster(self, source: str,
                out:str="res.txt",
                fd:str="models/FaceDetector.onnx",
                fr:str="models/FaceNet_vggface2_optmized.onnx",
                do_copy:bool=False,
                dst: str=None):
        if source:
            _cluster(source, fd, fr, out, do_copy, dst)
        else:
            print("Source dir is needed!")


if __name__ == "__main__":
    import fire
    fire.Fire(CLI())


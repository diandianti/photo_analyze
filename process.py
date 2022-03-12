#!/usr/bin/env python
# coding: utf-8

import os, sys
import collections
import numpy as np
from tqdm import tqdm
import signal
import shutil

import utils.file_tool as f_tool
from utils.file_tool import found_res, Res
from utils.path_tool import div_char
from utils.log import LOG
import utils.path_tool as p_tool
from nn.fd_with_lm import Fdlm
import nn.nn_run as nn_run
import utils.config as cfg
from nn.align_faces import warp_and_crop_face, get_reference_facial_points

if cfg.open_img_method == "cv2":
    from utils.file_tool import img_open_cv as img_open
    FD_CHANNEL_REVERSE = 0
    FR_CHANNEL_REVERSE = 1
else:
    from utils.file_tool import img_open_pil as img_open
    FD_CHANNEL_REVERSE = 1
    FR_CHANNEL_REVERSE = 0

temp_res = collections.namedtuple("TEMP_RES", ["id", "dis"])
g_exit = False
log = LOG(1)

def signal_handle(sig, frame):
    log.info("Get signal, save res and exit!")
    global g_exit
    g_exit = True

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

    return dst_img

def eucd_dis(a, b):
    return np.sqrt( ( (a - b) ** 2 ).sum() )


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

    ress = Res(outname, dst)
    idx = 0
    v_thd = cfg.fd.get("v_thd", 0.7)
    dis_thd = cfg.cluster.get("dis_thd", 0.8)

    global g_exit

    for img in tqdm(all_img):

        try:
            imga = img_open(img)
        except:
            log.err("Open image %s fail!"%img)
            continue

        bboxes, confs, lms = nn_fd.run(imga, FD_CHANNEL_REVERSE)

        embs = []
        for b,c,l in zip(bboxes, confs, lms):
            if c < v_thd:
                continue
            img_crop = align_face(imga, l, fr_input_size)
            emb = nn_fr.run(img_crop, FR_CHANNEL_REVERSE)
            embs.append(emb)

        for emb in embs:
            temp = []
            for id,v in ress.all_found.items():
                for e in v.embs:
                    dis = eucd_dis(emb, e)
                    if dis < dis_thd:
                        temp.append(temp_res(id, dis))

            if temp:
                temp.sort(key=lambda x: x.dis)
                target_id = temp[0].id
                ress.all_found[target_id].embs.append(emb)
                ress.all_found[target_id].imgs.append(img)
            else:
                idx += 1
                target_id = idx
                ress.all_found[target_id] = found_res(target_id, [emb], [img])

        if g_exit:
            break

    ress.save(outname)
    if do_copy:
        ress.copy_images(dst)


def get_reg_emb(all_reg, nn_fd, nn_fr):

    res_emb = []
    v_thd = cfg.fd.get("v_thd", 0.7)
    fr_input_size = nn_fr.input_size

    for p in tqdm(all_reg):

        id = p["path"].split(div_char)[-1]
        embs = []

        for f in p["files"]:
            img_p = os.path.join(p["path"], f)
            try:
                imga = img_open(img_p)
            except:
                log.err("Open image %s fail!" % img_p)
                continue

            bboxes, confs, lms = nn_fd.run(imga, FD_CHANNEL_REVERSE)
            one_embs = []
            for b, c, l in zip(bboxes, confs, lms):
                if c < v_thd:
                    continue
                img_crop = align_face(imga, l, fr_input_size)
                emb = nn_fr.run(img_crop, FR_CHANNEL_REVERSE)
                one_embs.append(emb)

            embs.extend(emb)

        if len(embs) == 0:
            continue

        res_emb.append({
            "id": id,
            "embs": embs
            })

    return res_emb

def _compare(not_path, konw_path, fd_model, fr_model, dst):

    # get all image have name
    all_konw = p_tool.get_all(konw_path, res_type="dict")
    # get all image no name
    all_not = p_tool.get_all(not_path)
    # init nn
    nn_fd = Fdlm(fd_model,
                 confidence_threshold=cfg.fd.get("confidence_threshold"),
                 top_k=cfg.fd.get("top_k"),
                 keep_top_k=cfg.fd.get("keep_top_k"),
                 nms_threshold=cfg.fd.get("nms_threshold"))
    nn_fr = nn_run.Fr(fr_model)

    _embf = f_tool.CacheTool(konw_path)
    tmp = _embf.load()
    if tmp:
        log.info("Get base embs from cache file!")
        res_emb = tmp
    else:
        log.info("Get base embs from fr run!")
        res_emb = get_reg_emb(all_konw, nn_fd, nn_fr)
        _embf.save(res_emb)

    fr_input_size = nn_fr.input_size
    v_thd = cfg.fd.get("v_thd", 0.7)
    dis_thd = cfg.compare.get("dis_thd", 0.75)
    global g_exit

    for one in tqdm(all_not):
        log.info("Process ... %s"%one)
        # try to open image
        try:
            imga = img_open(one)
        except:
            log.err("Open %s fail!"%one)
            continue

        # get all face emb in one image
        bboxes, confs, lms = nn_fd.run(imga, FD_CHANNEL_REVERSE)
        embs_in_image = []
        for b, c, l in zip(bboxes, confs, lms):
            if c < v_thd:
                continue
            img_crop = align_face(imga, l, fr_input_size)
            emb = nn_fr.run(img_crop, FR_CHANNEL_REVERSE)
            embs_in_image.append(emb)

        for ne in embs_in_image:
            sames = []
            for r in res_emb:
                for e in r["embs"]:
                    dis = eucd_dis(e, ne)
                    if dis < dis_thd:
                        sames.append(temp_res(r["id"], dis))

            if len(sames) == 0:
                continue

            sames.sort(key=lambda x: x.dis)
            target = sames[0]

            tar_dir = os.path.join(dst, target.id)
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)

            try:
                shutil.copy2(one, tar_dir)
            except:
                log.err("Copy file fail!")
                continue

        if g_exit:
            break

class CLI():

    def __init__(self):

        signal.signal(signal.SIGINT, signal_handle)
        if sys.platform.find("linux") == 0:
            signal.signal(signal.SIGKILL, signal_handle)
            signal.signal(signal.SIGQUIT, signal_handle)
            signal.signal(signal.SIGTERM, signal_handle)

    def cluster(self, source: str,
                out:str="res.txt",
                fd:str="models/FaceDetector.onnx",
                fr:str="models/FaceNet_vggface2_optmized.onnx",
                do_copy:bool=False,
                dst: str=None,
                log_level: int=1):
        log.set_level(log_level)
        if source:
            _cluster(source, fd, fr, out, do_copy, dst)
        else:
            log.err("Source dir is needed!")

    def compare(self, source: str,
                base: str,
                fd: str="models/FaceDetector.onnx",
                fr:str="models/FaceNet_vggface2_optmized.onnx",
                dst: str=None,
                log_level: int=1):
        log.set_level(log_level)
        copy_to = base
        if dst:
            copy_to = dst

        _compare(source, base, fd, fr, copy_to)


if __name__ == "__main__":
    import fire
    fire.Fire(CLI())


open_img_method="cv2" #or "pillow"

fd = {
    "v_thd": 0.7,
    "confidence_threshold" : 0.02,
    "top_k" : 5000,
    "keep_top_k" : 750,
    "nms_threshold": 0.4
}


cluster = {
    "dis_thd": 0.8
}

compare = {
    "dis_thd": 0.75
}
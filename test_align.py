import cv2
import numpy as np

from nn.fd_with_lm import Fdlm
from nn.align_faces import warp_and_crop_face, get_reference_facial_points

def align_face(img, lm: np.ndarray, output_size, idx):
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
    cv2.imwrite('{}_retinaface_aligned_{}x{}.jpg'.format(idx, output_size[0], output_size[1]), dst_img)

def test(img):
    img = cv2.imread(img)
    fd = Fdlm("./models/FaceDetector.onnx")
    bboxes, conf, lms = fd.run(img)

    count = 0
    for lm,c in zip(lms, conf):
        if c < 0.7:
            continue

        align_face(img, lm, (160, 160), count)
        count += 1

    fd.draw_img(img, bboxes, conf, lms)

if __name__ == "__main__":
    import fire
    fire.Fire(test)

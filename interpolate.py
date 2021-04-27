import math
import os
import math
import numpy as np
from typing import *

import torch
import clip
import numpy as np
from imageio import imsave

from clip_fft import to_valid_rgb, fft_image, slice_imgs, checkout, cvshow
from utils import pad_up_to, basename, file_list, img_list, img_read, txt_clean, plot_text

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def generate_interpolations(
    pt_dir: str,
    out_video_dir: str,
    out_video_name: str,
    scene_len: int = 180,
    fps: int = 25,
    size: List[str] = [720, 1280],
    decay: float = 1.,
    colors: float = 1.,
):
    pt_file_list = file_list(pt_dir, 'pt')
    num_frames = scene_len * fps

    for pt_idx in range(len(pt_file_list)):
        params1 = read_pt(pt_file_list[pt_idx])
        params2 = read_pt(pt_file_list[(pt_idx+1) % len(pt_file_list)])

        _params, image_f = fft_image([1, 3, *size], resume=params1, sd=1., decay_power=decay,)
        image_f = to_valid_rgb(image_f, colors=colors,)

        for frame_idx in range(num_frames):
            with torch.no_grad():
                img = image_f((params2 - params1) * math.sin(1.5708 * frame_idx/num_frames)**2)[0].permute(1,2,0)
                img = torch.clip(img*255, 0, 255).cpu().numpy().astype(np.uint8)

            imsave(os.path.join(out_video_dir, '%05d.jpg' % (pt_idx * num_frames + frame_idx)), img)

    os.system('ffmpeg -v warning -y -i %s\%%05d.jpg "%s.mp4"' % (out_video_dir, out_video_name))

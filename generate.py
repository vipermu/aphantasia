import os
import shutil
from typing import *

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import clip
from imageio import imsave

from utils import img_list, txt_clean
from progress_bar import ProgressBar
from arguments import get_args

img_norm = torchvision.transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(height, width):
    """Computes 2D spectrum frequencies."""
    y_freqs = np.fft.fftfreq(height)[:, None]
    # when we have an odd input dimension we need to keep one additional frequency and later cut off 1 pixel
    width_even_idx = (width + 1) // 2 if width % 2 == 1 else width // 2 + 1

    x_freqs = np.fft.fftfreq(width)[:width_even_idx]

    return np.sqrt(x_freqs * x_freqs + y_freqs * y_freqs)


def get_fft_img(
    spectrum_size: Union[List[int], Tuple[int]],
    std: float = 0.01,
    return_img_freqs=False,
):
    """
    """
    batch_size, num_channels, height, width = spectrum_size

    #NOTE: generate all possible freqs for the input image size
    img_freqs = rfft2d_freqs(height, width)

    #NOTE: 2 for imaginary and real components
    spectrum_shape = [
        batch_size,
        num_channels,
        *img_freqs.shape,
        2,
    ]

    fft_img = (torch.randn(*spectrum_shape) * std)

    if return_img_freqs:
        return fft_img, img_freqs
    else:
        return fft_img


def get_scale_from_img_freqs(
    img_freqs,
    decay_power,
):
    height, width = img_freqs.shape
    clamped_img_freqs = np.maximum(img_freqs, 1.0 / max(width, height))

    scale = 1.0 / clamped_img_freqs**decay_power
    scale *= np.sqrt(width * height)
    scale = torch.tensor(scale).float()[None, None, ..., None]

    return scale


def fft_to_rgb(
    fft_img,
    scale,
    img_size,
    shift=None,
    contrast=1.,
    decorrelate=True,
    device="cuda",
):
    num_channels = 3
    im = 2

    scaled_fft_img = scale * fft_img
    if shift is not None:
        scaled_fft_img += scale * shift

    image = torch.irfft(
        scaled_fft_img,
        im,
        normalized=True,
        signal_sizes=img_size,
    )
    image = image * contrast / image.std()  # keep contrast, empirical

    if decorrelate:
        colors = 1
        color_correlation_svd_sqrt = np.asarray([
            [0.26, 0.09, 0.02],
            [0.27, 0.00, -0.05],
            [0.27, -0.09, 0.03],
        ]).astype("float32")
        color_correlation_svd_sqrt /= np.asarray([
            colors,
            1.,
            1.,
        ])  # saturate, empirical

        max_norm_svd_sqrt = np.max(
            np.linalg.norm(color_correlation_svd_sqrt, axis=0))

        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

        image_permute = image.permute(0, 2, 3, 1).to(device)
        image_permute = torch.matmul(
            image_permute,
            torch.tensor(color_correlation_normalized.T).to(device),
        )

        image = image_permute.permute(0, 3, 1, 2)

    image = torch.sigmoid(image)

    return image


def checkout(
    img,
    fname=None,
):
    img = np.transpose(np.array(img)[:, :, :], (1, 2, 0))
    if fname is not None:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        imsave(fname, img)


def random_crop(
    img,
    num_crops,
    crop_size=224,
    normalize=True,
):
    def map(x, a, b):
        return x * (b - a) + a

    rnd_size = torch.rand(num_crops)
    rnd_offx = torch.clip(torch.randn(num_crops) * 0.2 + 0.5, 0., 1.)
    rnd_offy = torch.clip(torch.randn(num_crops) * 0.2 + 0.5, 0., 1.)

    img_size = img.shape[2:]
    min_img_size = min(img_size)

    sliced = []
    cuts = []
    for c in range(num_crops):
        current_crop_size = map(rnd_size[c], crop_size, min_img_size).int()

        offsetx = map(rnd_offx[c], 0, img_size[1] - current_crop_size).int()
        offsety = map(rnd_offy[c], 0, img_size[0] - current_crop_size).int()
        cut = img[:, :, offsety:offsety + current_crop_size,
                  offsetx:offsetx + current_crop_size]
        cut = F.interpolate(
            cut,
            (crop_size, crop_size),
            mode='bicubic',
            align_corners=False,
        )  # bilinear

        if normalize is not None:
            cut = img_norm(cut)

        cuts.append(cut)

    return torch.cat(cuts, axis=0)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _ = clip.load(args.model)
    print(f"Using model {args.model}")

    input_text = args.input_text
    print(f"Generating from '{input_text}'")

    out_name_list = []
    out_name_list.append(txt_clean(input_text))
    out_name = '-'.join(out_name_list)
    out_name += '-%s' % args.model if 'RN' in args.model.upper() else ''

    tempdir = os.path.join(args.out_dir, out_name)
    os.makedirs(tempdir, exist_ok=True)

    tokenized_text = clip.tokenize([input_text]).to(device).detach().clone()
    text_logits = clip_model.encode_text(tokenized_text)

    num_channels = 3
    spectrum_size = [args.batch_size, num_channels, *args.size]
    fft_img, img_freqs = get_fft_img(
        spectrum_size,
        std=0.01,
        return_img_freqs=True,
    )

    fft_img = fft_img.to(device)
    fft_img.requires_grad = True

    scale = get_scale_from_img_freqs(
        img_freqs=img_freqs,
        decay_power=args.decay,
    )

    scale = scale.to(device)

    shift = None
    if args.noise > 0:
        img_size = img_freqs.shape
        noise_size = (1, 1, *img_size, 1)
        shift = self.noise * torch.randn(noise_size, ).to(self.device)

    optimizer = torch.optim.Adam(
        [fft_img],
        args.lrate,
    )

    sign = -1

    pbar = ProgressBar(args.num_steps // args.save_freq)

    num_steps = args.num_steps
    num_crops = 200
    crop_size = 224

    for step in range(num_steps):
        loss = 0

        initial_img = fft_to_rgb(
            fft_img=fft_img,
            scale=scale,
            img_size=args.size,
            shift=shift,
            contrast=1.0,
            decorrelate=True,
            device=device,
        )

        crop_img_out = random_crop(
            initial_img,
            num_crops,
            crop_size,
            normalize=True,
        )
        img_logits = clip_model.encode_image(crop_img_out).to(device)
        tokenized_text = clip.tokenize([input_text]).to(device)
        text_logits = clip_model.encode_text(tokenized_text)

        loss += -torch.cosine_similarity(
            text_logits,
            img_logits,
            dim=-1,
        ).mean()

        torch.cuda.empty_cache()

        # if self.prog is True:
        #     lr_cur = lr + (step / self.steps) * (init_lr - lr)
        #     for g in self.optimizer.param_groups:
        #         g['lr'] = lr_cur

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.save_freq == 0:
            with torch.no_grad():
                img = fft_to_rgb(
                    fft_img=fft_img,
                    scale=scale,
                    img_size=args.size,
                    shift=shift,
                    contrast=1.0,
                    decorrelate=True,
                    device=device,
                )
                img = img.cpu().numpy()

            img_out_path = os.path.join(tempdir,
                                        '%04d.jpg' % (step // args.save_freq))
            checkout(
                img[0],
                img_out_path,
            )

            if pbar is not None:
                pbar.upd()

    os.system('ffmpeg -v warning -y -i %s\%%04d.jpg "%s.mp4"' %
              (tempdir, os.path.join(args.out_dir, out_name)))
    shutil.copy(
        img_list(tempdir)[-1],
        os.path.join(out_dir, '%s-%d.jpg' % (out_name, num_steps)))

    if args.save_pt is True:
        torch.save(fft_img, '%s.pt' % os.path.join(out_dir, out_name))


if __name__ == '__main__':
    main()

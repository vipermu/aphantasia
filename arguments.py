import argparse

model_list = ['ViT-B/32', 'RN50', 'RN50x4', 'RN101']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_text',
        default=None,
        help='input text',
    )
    parser.add_argument(
        '--micro_text',
        default=None,
        help='input text for small details',
    )
    parser.add_argument(
        '--substract_text',
        default=None,
        help='input text to subtract',
    )

    parser.add_argument(
        '--out_dir',
        default='_out',
    )
    parser.add_argument(
        '--size',
        default='1280-720',
        help='Output resolution',
    )
    parser.add_argument(
        '--save_freq',
        default=1,
        type=int,
        help='Saving step',
    )
    parser.add_argument(
        '--save_pt',
        action='store_true',
        help='Save FFT snapshots for further use',
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Batch size generation',
    )

    # training
    parser.add_argument(
        '--model',
        default='ViT-B/32',
        choices=model_list,
        help='Select CLIP model to use',
    )
    parser.add_argument(
        '--num_steps',
        default=200,
        type=int,
        help='Total iterations',
    )
    parser.add_argument(
        '--num_crops',
        default=200,
        type=int,
        help='Samples to evaluate',
    )
    parser.add_argument(
        '--lrate',
        default=0.05,
        type=float,
        help='Learning rate',
    )
    parser.add_argument(
        '--prog',
        action='store_true',
        help='Enable progressive lrate growth (up to double a.lrate)',
    )

    # tweaks
    parser.add_argument(
        '--contrast',
        default=1.,
        type=float,
    )
    parser.add_argument(
        '--colors',
        default=1.,
        type=float,
    )
    parser.add_argument(
        '--decay',
        default=1,
        type=float,
    )
    parser.add_argument(
        '--noise',
        default=0,
        type=float,
        help='Add noise to suppress accumulation',
    )  # < 0.05 ?
    parser.add_argument(
        '--sync',
        default=0,
        type=float,
        help='Sync output to input image',
    )
    parser.add_argument(
        '--invert',
        action='store_true',
        help='Invert criteria',
    )
    args = parser.parse_args()

    if args.size is not None:
        args.size = [int(s) for s in args.size.split('-')][::-1]

    if len(args.size) == 1:
        args.size = args.size * 2

    args.modsize = 288 if args.model == 'RN50x4' else 224

    return args
import jittor as jt
from jittor import init
from jittor import nn
from jittor import models

import argparse
from cam.layercam_vit import LayerCAM
from utils import basic_visualize, load_image, apply_transforms

from vision_transformer import vit_small_patch16_224

jt.flags.use_cuda = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='LayerCAM for Jittor')
    parser.add_argument('--img_path', type=str, default='./data/test/0/image_06734.jpg', help='Path of test image')
    parser.add_argument('--blocks_id', type=list, default=[1, 2, 5, 6], help='The cam generation blocks')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path of the checkpoint to load')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    input_image = load_image(args.img_path)
    input_ = apply_transforms(input_image)
    image_name = args.img_path.split('/')[-1].split('.')[0]

    # your own vit model
    # model = vit_small_patch16_224(num_classes=102)
    model = vit_small_patch16_224()
    if args.ckpt_path is not None:
        model.load(args.ckpt_path)
    optimizer = nn.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for block in args.blocks:
        model_dict = {
            'type': 'vision_transformer',
            'arch': model,
            'layer_name': block,
        }
        vit_layercam = LayerCAM(model_dict, optimizer)
        layercam_map = vit_layercam(input_)
        basic_visualize(
            input_.numpy(),
            layercam_map.numpy(),
            cmap='jet',
            save_path='./vis/img_{}_block_{}.png'.format(image_name, block))

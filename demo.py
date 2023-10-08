import glob
import os
import time

import cv2
import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
from collections import Counter


@torch.no_grad()
def run_demo(device_str, video, cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if video == 'true':
        video = 1
    if video == 'false':
        video = 0
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    # device = torch.device(cfg.MODEL.DEVICE)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_strs = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == 0:
        device_strs = 'cuda'
    if device_str == -1:
        device_strs = 'cpu'
    device = torch.device(device_strs)
    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device('cpu')

    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    video_capture = cv2.VideoCapture(0)
    video_writer = None

    # video = 1
    if video == 0:
        for i, image_path in enumerate(image_paths):
            start = time.time()
            image_name = os.path.basename(image_path)

            image = np.array(Image.open(image_path).convert("RGB"))
            height, width = image.shape[:2]
            images = transforms(image)[0].unsqueeze(0)
            load_time = time.time() - start

            start = time.time()
            result = model(images.to(device))[0]
            inference_time = time.time() - start

            result = result.resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            meters = ' | '.join(
                [
                    'objects {:02d}'.format(len(boxes)),
                    'names {}'.format(labels),
                    'confidence {}'.format(scores),
                    'load {:03d}ms'.format(round(load_time * 1000)),
                    'inference {:03d}ms'.format(round(inference_time * 1000)),
                    'FPS {}'.format(round(1.0 / inference_time))
                ]
            )
            print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

            drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
            Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
    if video == 1:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            start = time.time()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            images = transforms(image)[0].unsqueeze(0)
            load_time = time.time() - start

            start = time.time()
            result = model(images.to(device))[0]
            inference_time = time.time() - start

            result = result.resize((width, height)).to(cpu_device)
            result = result.numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            # list = [
            #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            #     'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            #     'tvmonitor']
            list = [
                '__background__', 'aid_120', 'fire', 'fire_119', 'first_aid', 'front_left', 'front_right', 'hazmat',
                'hazmat_110',
                'left', 'left_right', 'right']
            # number = int(labels)
            label_counts = Counter(labels)
            # label_str = ', '.join(
            # "{} ({})".format((list[i], count) + ('s' if count > 1 else '') for i, count in label_counts.items()))
            label_str = ', '.join(
                '{} {}'.format(count, list[i]) + ('s' if count > 1 else '') for i, count in label_counts.items())
            meters = ' | '.join(
                [
                    'objects {:02d}'.format(len(boxes)),
                    'names: {}'.format(label_str),
                    'load {:03d}ms'.format(round(load_time * 1000)),
                    'inference {:03d}ms'.format(round(inference_time * 1000)),
                    'FPS {}'.format(round(1.0 / inference_time))
                ]
            )
            print(meters)

            drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
            # if labels == 15:
            #     name = 'person'
            # print(name)
            cv2.imshow('Detection', drawn_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if video_writer is None:
                output_path = os.path.join(output_dir, 'output.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            video_writer.write(cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))

        video_capture.release()
        if video_writer is not None:
            video_writer.release()


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="configs/mobilenet_v2_ssd320_voc0712.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default='little/model_final.pth', help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.70)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str,
                        help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and coco.')
    parser.add_argument("--video", default="true", type=str,
                        help='true for video;false for images/image')
    parser.add_argument("--device", default="0", type=int,
                        help='0 for gpu;-1 for cpu')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(device_str=args.device,
             video=args.video,
             cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()

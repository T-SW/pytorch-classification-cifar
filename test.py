import os
import argparse
import torch
from timm.data import resolve_data_config
from utils.dataloaders import get_test_dataloader
from models.main import get_network

from PIL import Image
import glob
import numpy as np
import cv2
from torchvision import transforms


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str, help='net type')
    parser.add_argument('--data_dir', metavar='DIR', default=r"./test_img", help='path to dataset')  # C:\Users\Acer\Desktop\pj\Data\cifar10\test
    parser.add_argument('--weights', metavar='DIR', default=r"./checkpoint/resnet18-last.pth", help='the weights file you want to test')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('--num_classes', type=int, default=10, help='number for label classes')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or None')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for dataloader')

    parser.add_argument('--visual', default=False, help='visual or not')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [int(i) for i in args.device if i != ',']))
    net = get_network(args)

    data_config = resolve_data_config(vars(args))

    test_loader = get_test_dataloader(
        data_dir=args.data_dir,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=4,
        batch_size=args.batch_size,
    )

    test_num = len(test_loader.dataset)

    net.load_state_dict(torch.load(args.weights)['model_state_dict'])
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    if not args.visual:
        with torch.no_grad():
            for n_iter, (image, label) in enumerate(test_loader):
                print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

                if args.device:
                    image = image.cuda()
                    label = label.cuda()

                output = net(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                # compute top 5
                correct_5 += correct[:, :5].sum()

                # compute top1
                correct_1 += correct[:, :1].sum()

        print()
        print("Top 1 err: ", 1 - correct_1 / test_num)
        print("Top 5 err: ", 1 - correct_5 / test_num)
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

    else:
        im_test_list = glob.glob(args.data_dir + "\*\*.jpg")
        np.random.shuffle(im_test_list)

        label_name = ["airplane",
                      "automobile",
                      "bird",
                      "cat",
                      "deer",
                      "dog",
                      "frog",
                      "horse",
                      "ship",
                      "truck"]

        test_transform = transforms.Compose([
            transforms.CenterCrop((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2033, 0.1994, 0.2010)),
        ])

        for im_path in im_test_list:
            net.eval()
            im_data = Image.open(im_path)

            inputs = test_transform(im_data)

            inputs = torch.unsqueeze(inputs, dim=0)

            inputs = inputs.cuda()

            outputs = net.forward(inputs)

            _, pred = torch.max(outputs.data, dim=1)

            print(label_name[pred.cpu().numpy()[0]])

            img = np.asarray(im_data)

            img = img[:, :, [1, 2, 0]]  # RGB转为BGR

            img = cv2.resize(img, (300, 300))

            cv2.imshow("im", img)
            cv2.waitKey(0)

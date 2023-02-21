import sys


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=args.num_classes)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes=args.num_classes)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=args.num_classes)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_classes=args.num_classes)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_classes=args.num_classes)
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(num_classes=args.num_classes)
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(num_classes=args.num_classes)
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(num_classes=args.num_classes)
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(num_classes=args.num_classes)
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(num_classes=args.num_classes)
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(num_classes=args.num_classes)
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(num_classes=args.num_classes)
    elif args.net == 'xception':
        from models.xception import xception
        net = xception(num_classes=args.num_classes)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=args.num_classes)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=args.num_classes)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=args.num_classes)
    elif args.net == 'resnet50ch':
        from models.resnet50ch import resnet50ch
        net = resnet50ch(num_classes=args.num_classes)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=args.num_classes)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=args.num_classes)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(num_classes=args.num_classes)
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(num_classes=args.num_classes)
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(num_classes=args.num_classes)
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(num_classes=args.num_classes)
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(num_classes=args.num_classes)
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_classes=args.num_classes)
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(num_classes=args.num_classes)
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(num_classes=args.num_classes)
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(num_classes=args.num_classes)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(num_classes=args.num_classes)
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(class_num=args.num_classes)
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(class_num=args.num_classes)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(num_classes=args.num_classes)
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(num_classes=args.num_classes)
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56(num_classes=args.num_classes)
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92(num_classes=args.num_classes)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(num_classes=args.num_classes)
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(num_classes=args.num_classes)
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(num_classes=args.num_classes)
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(num_classes=args.num_classes)
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(num_classes=args.num_classes)
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet(num_classes=args.num_classes)
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18(num_classes=args.num_classes)
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34(num_classes=args.num_classes)
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50(num_classes=args.num_classes)
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101(num_classes=args.num_classes)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.device:    # use_gpu
        net = net.cuda()

    return net
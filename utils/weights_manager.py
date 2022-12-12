import torch


def unet_manager(in_channels, classes):
    try:
        from unet import unet
        bilinear = False
        weights_path = '../converter/model_forge/f2e4a3a6-f9d7-49fc-a9da-79fb325c3899/58d3ebdb-ba6c-4bec-a9fb-66195abb7f00/weights/best_dice.pth'
        checkpoint = torch.load(weights_path)
        weight = checkpoint['net']['outc.conv.weight']
        net = unet.UNet(n_channels=in_channels, n_classes=classes, bilinear=bilinear)

        print(weight.shape[0])
        if weight.shape[0] != classes:
            # manage weights
            # class added or removed?
            if weight.shape[0] > classes:
                # class removed - detect removed class and remove corresponded neuron weight
                pass
            else:
                # class added - add extra neurons with weights and biases
                pass

        # try to load weights
        net.load_state_dict(managed_weights)

    except Exception as e:
        print(e)


def yolov5_manager(weights_path):
    # from yolov5 import *
    pass


def yolov7_manager(weights_path):
    pass


def main():
    in_channels, classes = 3, 3
    unet_manager(in_channels, classes)
    pass


if __name__ == '__main__':
    main()

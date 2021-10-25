import torch
import torchvision
import struct


if __name__ == '__main__':
    r3d = torchvision.models.video.r3d_18(pretrained=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    r3d = r3d.to(device)
    with open('r3d.wts', 'w') as f:
        f.write('{}\n'.format(len(r3d.state_dict().keys())))
        for k, v in r3d.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

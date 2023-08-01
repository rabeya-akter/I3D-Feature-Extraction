import os

import numpy as np
import torch
import cv2
import math

from models.pytorch_i3d import InceptionI3d

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_rgb_frames(image_dir, vid, start, num, desired_channel_order='rgb'):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, "images" + str(i).zfill(4) + '.png'))

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_all_rgb_frames_from_folder(folder, desired_channel_order='rgb'):
    frames = []

    for fn in sorted(os.listdir(folder), key=lambda x: int(x[6:6+4])):
        img = cv2.imread(os.path.join(folder, fn))
        img = cv2.resize(img, dsize=(224, 224))

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_rgb_frames_from_video(vid_path, start, num, size=(224, 224), desired_channel_order='rgb'):
    vidcap = cv2.VideoCapture(vid_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = vidcap.read()

        w, h, c = img.shape

        """
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1
        """

        img = cv2.resize(img, (224,224))
        img = (img / 255.) * 2 - 1

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def extract_features_fullvideo(model, inp, framespan, stride):
    rv = []

    indices = list(range(len(inp)))
    groups = []
    for ind in indices:

        if ind % stride == 0:
            groups.append(list(range(ind, ind+framespan)))

    for g in groups:
        # numpy array indexing will deal out-of-index case and return only till last available element
        frames = inp[g[0]: min(g[-1]+1, inp.shape[0])]

        num_pad = 9 - len(frames)
        if num_pad > 0:
            pad = np.tile(np.expand_dims(frames[-1], axis=0), (num_pad, 1, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.transpose([3, 0, 1, 2])

        ft = _extract_features(model, frames)

        #changes to npy
        ft=ft.numpy()
        ft=np.squeeze(ft)

        rv.append(ft)



    return rv


def _extract_features(model, frames):
    inputs = torch.from_numpy(frames)

    inputs = inputs.unsqueeze(0)

    inputs = inputs.cuda()
    with torch.no_grad():
        ft = model.extract_features(inputs)
    ft = ft.squeeze(-1).squeeze(-1)[0].transpose(0, 1)

    ft = ft.cpu()

    return ft


def run(weight, frame_roots, outroot, inp_channels='rgb'):
    frame_dirs = []

    for root in frame_roots:
        paths = sorted(os.listdir(root))

        frame_dirs.extend([os.path.join(root, path) for path in paths])

    # ===== setup models ======
    i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(2000)
    #i3d.replace_logits(1232)

    print('loading weights {}'.format(weight))
    # i3d.load_state_dict(torch.load(weight))


    #i3d.load_state_dict(torch.load(weight)['ckpt'])
    i3d.load_state_dict(torch.load(weight))



    i3d.cuda()
    # i3d = nn.DataParallel(i3d)
    #i3d.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print('feature extraction starts.')
    i3d.train(False)  # Set model to evaluate mode

    # ===== extract features ======
    # for framespan, stride in [(4, 2), (16, 8), (32, 16)]:
    #for framespan, stride in [(16, 2), (12, 2), (8, 2)]:
    for framespan, stride in [(16, 2)]:
    # for framespan, stride in [(16, 8), (32, 16), (64, 32)]:

        outdir = os.path.join(outroot, 'span={}_stride={}'.format(framespan, stride))

        if not os.path.exists(outdir):
            os.makedirs(outdir)


        for ind, dir in enumerate(frame_dirs):
            #out_path = os.path.join(outdir, os.path.basename(dir)) + '.pt'
            out_path = os.path.join(outdir, os.path.basename(dir))[:-4]


            if os.path.exists(out_path):
                print('{} exists, continue'.format(out_path))
                continue

            #frames = load_all_rgb_frames_from_folder(dir, inp_channels)
            frames = load_rgb_frames_from_video(dir,0,2000, inp_channels)
            features = extract_features_fullvideo(i3d, frames, framespan, stride)

            #changes for npy
            features=np.array(features)


            if ind % 1 == 0:
                print(ind, dir, len(features), features[0].shape)

            #torch.save(features, os.path.join(outdir, os.path.basename(dir)) + '.pt')
            np.save(out_path + '.npy',features)


if __name__ == "__main__":
    weight = 'checkpoints/archive/nslt_2000_065538_0.514762.pt'

    # ======= Extract Features for PHEOENIX-2014-T ========
    frame_roots = [
        "/media/rmedu/New Volume/Rabeya Akter/How2Sign Dataset/val_rgb_front_clips/raw_videos"
    ]

    # out = '/home/dongxu/Dongxu/workspace/translation/data/PHOENIX-2014-T/features/i3d-features'
    out = '/media/rmedu/New Volume/Rabeya Akter/How2Sign Dataset/i3d-features/val'

    run(weight, frame_roots, out, 'rgb')

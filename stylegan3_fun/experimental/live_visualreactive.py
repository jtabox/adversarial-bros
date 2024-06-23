import os
from typing import List, Union, Optional, Tuple, Type
import click
import time
import copy

from stylegan3_fun import dnnlib
from stylegan3_fun.torch_utils import gen_utils

import numpy as np

import cv2
import imutils
import PIL.Image

import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
from torchvision import transforms

from stylegan3_fun import legacy

from stylegan3_fun.network_features import VGG16FeaturesNVIDIA

import mediapipe as mp


# ----------------------------------------------------------------------------


def parse_height(s: str = None) -> Union[int, Type[None]]:
    """Parse height argument."""
    if s is not None:
        if s == 'max':
            return s
        else:
            return int(s)
    return None


# ----------------------------------------------------------------------------


# TODO: Analyze latent space/variant to the proposed PCA https://openreview.net/pdf?id=SlzEll3EsKv
# TODO: Add hand tracking/normalization here: https://github.com/caillonantoine/hand_osc/blob/master/detect.py

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename: can be URL, local file, or the name of the model in torch_utils.gen_utils.resume_specs', required=True)
@click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# Synthesis options (feed a list of seeds or give the projected w to synthesize)
@click.option('--seed', type=click.INT, help='Random seed to use for static synthesized image', default=0, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.6, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
@click.option('--mirror', is_flag=True, help='Mirror the synthesized image')
@click.option('--demo-height', type=int, help='Height of the demo window', default=360, show_default=True)
@click.option('--demo-width', type=int, help='Width of the demo window', default=None, show_default=True)
@click.option('--only-synth', is_flag=True, help='Only show the synthesized image and save it to disk')
@click.option('--layer', type=str, help='Layer of the pre-trained VGG16 to use as the feature extractor', default='conv4_1', show_default=True)
# Mediapipe options
@click.option('--hands', 'hand_tracking', type=bool, help='Use hand tracking', default=True, show_default=True)
@click.option('--face', 'face_tracking', type=bool, help='Use face tracking', default=False, show_default=True)
@click.option('--body', 'body_tracking', type=bool, help='Use body tracking', default=False, show_default=True)
# How to set the fake dlatent
@click.option('--v0', is_flag=True, help='Average the features of VGG and use a static dlatent to do style-mixing')
@click.option('--v1', is_flag=True, help='Separate the input image into regions for coarse, middle, and fine layers for style-mixing')
@click.option('--v2', is_flag=True, help='Manipulate the input to the Generator (StyleGAN2 and 3)')
@click.option('--v3', is_flag=True, help='Latent mirror. Warning, should be used with low-resolution models (e.g., 16x16)')
# TODO: intermediate layers?
# Video options
@click.option('--display-height', type=parse_height, help="Height of the display window; if 'max', will use G.img_resolution", default=None, show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
@click.option('--fps', type=click.IntRange(min=1), help='Save the video with this framerate.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with `ffmpeg-python` (same resolution, lower file size)')
# Extra parameters
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(),
                                                                                                                            '../out', 'videos'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='live_visual-reactive', show_default=True)
@click.option('--verbose', is_flag=True, help='Print FPS of the live interpolation ever second; plot the detected hands for `--v2`')
def live_visual_reactive(
        ctx,
        network_pkl: str,
        device: Optional[str],
        cfg: str,
        seed: int,
        truncation_psi: float,
        class_idx: int,
        noise_mode: str,
        new_center: Union[int, str],
        mirror: bool,
        demo_height: int,
        demo_width: int,
        only_synth: bool,
        layer: str,
        hand_tracking: bool,
        face_tracking: bool,
        body_tracking: bool,
        v0: bool,
        v1: bool,
        v2: bool,
        v3: bool,
        display_height: Optional[int],
        anchor_latent_space: bool,
        fps: int,
        compress: bool,
        outdir: str,
        description: str,
        verbose: Optional[bool]):
    """Live Visual-Reactive interpolation. A camera/webcamera is needed to be accessed by OpenCV."""
    # Set device; GPU is recommended
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')

    if v0 or v1:
        # Load the feature extractor; here, VGG16
        print('Loading VGG16 and its features...')
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

        vgg16_features = VGG16FeaturesNVIDIA(vgg16).requires_grad_(False).to(device)
        del vgg16

    # If model name exists in the gen_utils.resume_specs dictionary, use it instead of the full url
    try:
        network_pkl = gen_utils.resume_specs[cfg][network_pkl]
    except KeyError:
        # Otherwise, it's a local file or an url
        pass

    print('Loading Generator...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)  # type: ignore

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Warm up the Generator
    ws = G.mapping(z=torch.randn(1, 512, device=device), c=None, truncation_psi=1.0)
    _ = G.synthesis(ws[:1])

    # Label, in case it's a class-conditional model
    class_idx = gen_utils.parse_class(G, class_idx, ctx)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Recenter the latent space, if specified
    if new_center is None:
        w_avg = G.mapping.w_avg
    else:
        new_center, new_center_value = new_center
        # We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
        if isinstance(new_center_value, int):
            w_avg = gen_utils.get_w_from_seed(G, device, new_center_value,
                                              truncation_psi=1.0)  # We want the pure dlatent
        elif isinstance(new_center_value, np.ndarray):
            w_avg = torch.from_numpy(new_center_value).to(device)
        else:
            ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

    # Set up the video capture dimensions
    height = demo_height
    width = int(4.0/3*demo_height) if demo_width is None else demo_width
    sheight = int(height)
    swidth = sheight

    # Set display size. If none specified or user selects 'max', will use image resolution of the network
    display_height = G.img_resolution if (display_height is None or display_height == 'max') else display_height

    # Fix a dlatent to do style mixing
    static_w = G.mapping(torch.randn(1, G.z_dim, device=device, generator=torch.Generator(device=device).manual_seed(seed)), c=None)

    cam = cv2.VideoCapture(0)
    idx = 0

    start_time = time.time()
    x = 1  # displays the frame rate every 1 second if verbose is True
    counter = 0
    starting = True  # Initialize some default values only one time
    recording_flag = False

    # Preprocess each image for VGG16
    preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    if v0 or v1:
        while cam.isOpened():
            # read frame
            idx += 1
            ret_val, img = cam.read()
            img = imutils.resize(img, height=height)
            if mirror:
                img = cv2.flip(img, 1)
            img = np.array(img).transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0).float().to(device)

            frame = preprocess(img).to(device)
            fake_z = vgg16_features.get_layers_features(frame, layers=[layer])[0]
            
            # v0 
            if v0:
                fake_z = fake_z.view(1, 512, -1).mean(2)

                # Perform EMA with previous fake_z
                if counter == 0:
                    prev_fake_z = fake_z
                # Do EMA
                fake_z = 0.2 * prev_fake_z + 0.8 * fake_z
                prev_fake_z = fake_z

                fake_w = gen_utils.z_to_dlatent(G, fake_z, label, truncation_psi)

                # Do style mixing
                fake_w[:, 4:] = static_w[:, 4:]

            # v1
            elif v1:
                _n, _c, h, w = fake_z.shape
                
                # Separate into coarse/middle/fine according to areas
                coarse_fake_z = fake_z[:, :, :h//2, :]
                middle_fake_z = fake_z[:, :, h//2:, :w//2]
                fine_fake_z = fake_z[:, :, h//2:, w//2:]

                # Convert them to the expected shape (each region will be their own latent)
                coarse_fake_z = coarse_fake_z.reshape(1, G.z_dim, -1).mean(2)
                middle_fake_z = middle_fake_z.reshape(1, G.z_dim, -1).mean(2)
                fine_fake_z = fine_fake_z.reshape(1, G.z_dim, -1).mean(2)

                # Get the respective dlatents
                coarse_fake_w = gen_utils.z_to_dlatent(G, coarse_fake_z, label, 1.0)
                middle_fake_w = gen_utils.z_to_dlatent(G, middle_fake_z, label, 1.0)
                fine_fake_w = gen_utils.z_to_dlatent(G, fine_fake_z, label, 1.0)
                fake_w = torch.cat([coarse_fake_w[:, :4], middle_fake_w[:, 4:8], fine_fake_w[:, 8:]], dim=1)  # [1, G.num_ws, G.z_dim]

                # Perform EMA with previous fake_w
                if counter == 0 and starting:
                    prev_fake_w = fake_w
                    starting = False
                # Do EMA
                fake_w = 0.4 * prev_fake_w + 0.6 * fake_w
                prev_fake_w = fake_w

            # Set images to expected data type
            img = img.clamp(0, 255).data[0].cpu().numpy()
            img = img.transpose(1, 2, 0).astype('uint8')

            simg = gen_utils.w_to_img(G, fake_w, noise_mode, w_avg, truncation_psi)[0]
            simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
            
            # display
            if not only_synth:
                display_width = int(4/3*display_height)
                # Resize input image from the camera
                img = cv2.resize(img, (display_width, display_height))
                # Resize accordingly the synthesized image
                simg = cv2.resize(simg, (display_height, display_height), interpolation=cv2.INTER_CUBIC)
                img = np.concatenate((img, simg), axis=1)
                cv2.imshow('Visuorreactive Demo', img)
            else:
                # Resize the synthesized image to the desired display height/width
                simg = cv2.resize(simg, (display_height, display_height))
                cv2.imshow('Visuorreactive Demo - Only Synth Image', simg)

            counter += 1

            # FPS counter
            if (time.time() - start_time) > x and verbose:
                print(f"FPS: {counter / (time.time() - start_time):0.2f}")
                counter = 0
                start_time = time.time()

            # ESC to quit; SPACE to start recording
            key = cv2.waitKey(1)

            if key == 27:
                break
            elif key == 32:
                # Transition from not recording to recording
                if not recording_flag:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    w = width + swidth if not only_synth else G.img_resolution
                    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, height))
                    recording_flag = True
                else:
                    recording_flag = False
                    out.release()

            if recording_flag:
                out.write(img)

        cam.release()
        cv2.destroyAllWindows()

    elif v2:
        # TODO: Clean this, this is a mess
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        assert sum([hand_tracking, face_tracking, body_tracking]) == 1, 'Only one tracking method can be used at a time!'

        mp_hands = mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # We will use one of hands, face, or body tracking. Find the first true condition and its corresponding
        # context manager. If all conditions are False, default to None
        conditions = [(hand_tracking, mp_hands), (face_tracking, mp_face_mesh), (body_tracking, mp_pose)]
        tracking, mp_tracking = next(((condition, context_manager) for condition, context_manager in conditions if condition), (None, None))

        # Loop and network setup
        # Generate a loop of images
        num_frames = 900
        shape = [num_frames, 1, G.z_dim]
        # Generate a loop of images
        all_latents = np.random.RandomState(seed).randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[3.0 * 30, 0, 0], mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))
        all_latents = torch.from_numpy(all_latents).to(device)

        c = 0

        if hasattr(G.synthesis, 'b4'):
            model_type = 'stylegan2'
            const_input = copy.deepcopy(G.synthesis.b4.const).cpu().numpy()
            const_input_interpolation = np.random.randn(num_frames, *const_input.shape).astype(np.float32)  # [num_frames, G.w_dim, 4, 4]
            const_input_interpolation = scipy.ndimage.gaussian_filter(const_input_interpolation, sigma=[fps, 0, 0, 0], mode='wrap')
            const_input_interpolation /= np.sqrt(np.mean(np.square(const_input_interpolation))) / 2
        elif hasattr(G.synthesis, 'input'):
            model_type = 'stylegan3'

        if mp_tracking is not None:
            # Use the context manager
            with mp_tracking as f:
                counter = 0

                while cam.isOpened():
                    success, image = cam.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        continue

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = f.process(image)

                    # Get the hand rotation w.r.t. the landmarks (concretely, the wrist [0] and middle finger [9]
                    # will define the Y axis; the x-axis will be 90 degrees counter-clockwise from it)
                    # Set the vertical direction as 0.0
                    if results.multi_hand_landmarks:
                        # Just the first hand
                        hand = results.multi_hand_landmarks[-1]

                        base = hand.landmark[0]
                        middle = hand.landmark[9]
                        dx = middle.x - base.x
                        dy = middle.y - base.y
                        angle = - np.pi / 2 - np.arctan2(dy,
                                                         dx)  # Set s.t. line(palm, middle finger) is the vertical axis

                        # Let's get the position in x and y of the center of the whole hand
                        # Calculate the center of the hand (average of all landmarks)
                        # Set the center of the image as the origin
                        x = 0.0
                        y = 0.0
                        z = 0.0
                        area_points = []
                        for idx, landmark in enumerate(hand.landmark):
                            x += landmark.x
                            y += landmark.y
                            z += landmark.z
                            if idx in range(0, 21, 4):
                                area_points.append([landmark.x, landmark.y])
                        x /= len(hand.landmark)
                        y /= len(hand.landmark)
                        z /= len(hand.landmark)

                        x -= 0.5
                        y -= 0.5

                        # Calculate the distance to the origin
                        dist = np.sqrt(x ** 2 + y ** 2)
                        # Normalize it
                        dist *= 4 * 2 ** 0.5  # Max distance is 1/sqrt(2)

                        # Get the area of the hand enclosed between the 5 fingers and the wrist
                        # We will use the trapezoidal rule to approximate the area
                        hand_area = 0.0
                        for i in range(len(area_points) - 1):
                            hand_area += (area_points[i][0] - area_points[i + 1][0]) * (
                                        area_points[i][1] + area_points[i + 1][1])
                        hand_area += (area_points[-1][0] - area_points[0][0]) * (area_points[-1][1] + area_points[0][1])
                        hand_area = abs(hand_area) / 2


                    else:
                        # EMAs toward zero
                        angle = 0.0 if starting else prev_angle * 0.9
                        x = 0.0 if starting else prev_x * 0.9
                        y = 0.0 if starting else prev_y * 0.9
                        z = 0.0 if starting else prev_z * 0.9
                        dist = 0.0 if starting else prev_dist * 0.9
                        hand_area = 0.0 if starting else prev_hand_area * 0.9

                    if counter == 0 and starting:
                        prev_angle = angle
                        prev_x = x
                        prev_y = y
                        prev_z = z
                        prev_dist = dist
                        prev_hand_area = hand_area
                        starting = False

                    # ema these values
                    angle = 0.2 * prev_angle + 0.8 * angle
                    prev_angle = angle

                    x = 0.2 * prev_x + 0.8 * x
                    prev_x = x
                    y = 0.2 * prev_y + 0.8 * y
                    prev_y = y
                    z = 0.2 * prev_z + 0.8 * z
                    prev_z = z

                    dist = 0.2 * prev_dist + 0.8 * dist
                    prev_dist = dist

                    hand_area = 0.2 * prev_hand_area + 0.8 * hand_area
                    prev_hand_area = hand_area

                    # FPS and angle
                    if (time.time() - start_time) > x and verbose:
                        print(f'[{c % num_frames} / {num_frames}] FPS: {counter / (time.time() - start_time):0.2f}, '
                              f'Angle (rad): {angle:.3f}, Hand Center: ({x:.3f}, {y:.3f}, {z:.3f}), Distance: {dist:.3f}, Area: {hand_area:.3f}')
                        counter = 0
                        start_time = time.time()

                    if hasattr(G.synthesis, 'input'):
                        # Rotate and translate the image
                        m = gen_utils.make_affine_transform(None, angle=angle, translate_x=x, translate_y=-y,
                                                            scale_x=1 + 2 * z, scale_y=1 + 2 * z)
                        m = np.linalg.inv(m)
                        # Finally, we pass the matrix to the generator
                        G.synthesis.input.transform.copy_(torch.from_numpy(m))

                    elif hasattr(G.synthesis, 'b4'):
                        G.synthesis.b4.const.copy_(torch.from_numpy(
                            (1 - dist) * const_input + const_input_interpolation[c % num_frames] * dist))

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # Replace image with a white background
                    image[:] = (255, 255, 255)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                    # Get the latent vectors
                    latent = all_latents[c % num_frames]
                    c += 1

                    # Synthesize the image
                    simg = gen_utils.z_to_img(G=G, latents=latent, label=label,
                                              truncation_psi=truncation_psi, noise_mode=noise_mode)[0]
                    simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)

                    # display
                    if not only_synth:
                        # Let's horizontally concatenate the input image and the synthesized image
                        # Resize input image from the camera
                        img_height = display_height
                        img_width = int(img_height * width / height)
                        img = cv2.resize(image, (img_width, img_height))
                        w, h = img_width, img_height

                        # Resize accordingly the synthesized image
                        simg = cv2.resize(simg, (display_height, display_height), interpolation=cv2.INTER_LINEAR)
                        w += display_height

                        # Concatenate and show the images
                        img = np.concatenate((simg, img), axis=1)
                        cv2.imshow('Visuorreactive Demo', img)

                        # display_width = int(4/3*display_height)
                        # Resize input image from the camera
                        # img = cv2.resize(image, (display_width, display_height))
                        # w, h = display_width, display_height
                        # Resize accordingly the synthesized image
                        # simg = cv2.resize(simg, (display_height, display_height), interpolation=cv2.INTER_CUBIC)
                        # w += h
                        # img = np.concatenate((img, simg), axis=1)
                        # cv2.imshow('Visuorreactive Demo', img)
                    else:
                        # Resize the synthesized image to the desired display height/width
                        simg = cv2.resize(simg, (display_height, display_height))
                        w, h = display_height, display_height
                        cv2.imshow('Visuorreactive Demo - Only Synth Image', simg)

                    counter += 1

                    key = cv2.waitKey(1)
                    # User presses 'ESC' to exit
                    if key == 27:
                        break
                    elif key == 32:
                        # Transition from not recording to recording
                        if not recording_flag:
                            print('Recording started')
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(f'visualreactive_input_v2_{model_type}.mp4', fourcc, fps, (w, h))
                            recording_flag = True
                        else:
                            print('Recording stopped')
                            recording_flag = False
                            out.release()

                    if recording_flag:
                        out.write(img)
                cam.release()
        else:
            print("No tracking method selected!")

        # with mp_hands as hands:
            # counter = 0
            #
            # while cam.isOpened():
            #     success, image = cam.read()
            #     if not success:
            #         print("Ignoring empty camera frame.")
            #         continue
            #
            #     # To improve performance, optionally mark the image as not writeable to
            #     # pass by reference.
            #     image.flags.writeable = False
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     results = hands.process(image)
            #
            #     # Get the hand rotation w.r.t. the landmarks (concretely, the wrist [0] and middle finger [9]
            #     # will define the Y axis; the x-axis will be 90 degrees counter-clockwise from it)
            #     # Set the vertical direction as 0.0
            #     if results.multi_hand_landmarks:
            #         # Just the first hand
            #         hand = results.multi_hand_landmarks[-1]
            #
            #         base = hand.landmark[0]
            #         middle = hand.landmark[9]
            #         dx = middle.x - base.x
            #         dy = middle.y - base.y
            #         angle = - np.pi / 2 - np.arctan2(dy, dx)  # Set s.t. line(palm, middle finger) is the vertical axis
            #
            #         # Let's get the position in x and y of the center of the whole hand
            #         # Calculate the center of the hand (average of all landmarks)
            #         # Set the center of the image as the origin
            #         x = 0.0
            #         y = 0.0
            #         z = 0.0
            #         area_points = []
            #         for idx, landmark in enumerate(hand.landmark):
            #             x += landmark.x
            #             y += landmark.y
            #             z += landmark.z
            #             if idx in range(0, 21, 4):
            #                 area_points.append([landmark.x, landmark.y])
            #         x /= len(hand.landmark)
            #         y /= len(hand.landmark)
            #         z /= len(hand.landmark)
            #
            #         x -= 0.5
            #         y -= 0.5
            #
            #         # Calculate the distance to the origin
            #         dist = np.sqrt(x ** 2 + y ** 2)
            #         # Normalize it
            #         dist *= 4*2 ** 0.5  # Max distance is 1/sqrt(2)
            #
            #         # Get the area of the hand enclosed between the 5 fingers and the wrist
            #         # We will use the trapezoidal rule to approximate the area
            #         hand_area = 0.0
            #         for i in range(len(area_points) - 1):
            #             hand_area += (area_points[i][0] - area_points[i + 1][0]) * (area_points[i][1] + area_points[i + 1][1])
            #         hand_area += (area_points[-1][0] - area_points[0][0]) * (area_points[-1][1] + area_points[0][1])
            #         hand_area = abs(hand_area) / 2
            #
            #
            #     else:
            #         # EMAs toward zero
            #         angle = 0.0 if starting else prev_angle * 0.9
            #         x = 0.0 if starting else prev_x * 0.9
            #         y = 0.0 if starting else prev_y * 0.9
            #         z = 0.0 if starting else prev_z * 0.9
            #         dist = 0.0 if starting else prev_dist * 0.9
            #         hand_area = 0.0 if starting else prev_hand_area * 0.9
            #
            #     if counter == 0 and starting:
            #         prev_angle = angle
            #         prev_x = x
            #         prev_y = y
            #         prev_z = z
            #         prev_dist = dist
            #         prev_hand_area = hand_area
            #         starting = False
            #
            #     # ema these values
            #     angle = 0.2 * prev_angle + 0.8 * angle
            #     prev_angle = angle
            #
            #     x = 0.2 * prev_x + 0.8 * x
            #     prev_x = x
            #     y = 0.2 * prev_y + 0.8 * y
            #     prev_y = y
            #     z = 0.2 * prev_z + 0.8 * z
            #     prev_z = z
            #
            #     dist = 0.2 * prev_dist + 0.8 * dist
            #     prev_dist = dist
            #
            #     hand_area = 0.2 * prev_hand_area + 0.8 * hand_area
            #     prev_hand_area = hand_area
            #
            #     # FPS and angle
            #     if (time.time() - start_time) > x and verbose:
            #         print(f'[{c % num_frames} / {num_frames}] FPS: {counter / (time.time() - start_time):0.2f}, '
            #               f'Angle (rad): {angle:.3f}, Hand Center: ({x:.3f}, {y:.3f}, {z:.3f}), Distance: {dist:.3f}, Area: {hand_area:.3f}')
            #         counter = 0
            #         start_time = time.time()
            #
            #     if hasattr(G.synthesis, 'input'):
            #     # Rotate and translate the image
            #         m = gen_utils.make_affine_transform(None, angle=angle, translate_x=x, translate_y=-y,
            #                                             scale_x=1+2*z, scale_y=1+2*z)
            #         m = np.linalg.inv(m)
            #         # Finally, we pass the matrix to the generator
            #         G.synthesis.input.transform.copy_(torch.from_numpy(m))
            #
            #     elif hasattr(G.synthesis, 'b4'):
            #         G.synthesis.b4.const.copy_(torch.from_numpy((1 - dist) * const_input + const_input_interpolation[c % num_frames] * dist))
            #
            #     # Draw the hand annotations on the image.
            #     image.flags.writeable = True
            #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #     # Replace image with a white background
            #     image[:] = (255, 255, 255)
            #     if results.multi_hand_landmarks and verbose:
            #         for hand_landmarks in results.multi_hand_landmarks:
            #             mp_drawing.draw_landmarks(
            #                 image,
            #                 hand_landmarks,
            #                 mp_hands.HAND_CONNECTIONS,
            #                 mp_drawing_styles.get_default_hand_landmarks_style(),
            #                 mp_drawing_styles.get_default_hand_connections_style())
            #
            #     # Get the latent vectors
            #     latent = all_latents[c % num_frames]
            #     c += 1
            #
            #     # Synthesize the image
            #     simg = gen_utils.z_to_img(G=G, latents=latent, label=label,
            #                               truncation_psi=truncation_psi, noise_mode=noise_mode)[0]
            #     simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
            #
            #     # display
            #     if not only_synth:
            #         # Let's vertically concatenate the input image and the synthesized image
            #         # Resize input image from the camera
            #         img_width = display_height
            #         img_height = int(img_width * height / width)
            #         img = cv2.resize(image, (img_width, img_height))
            #         w, h = img_width, img_height
            #
            #         # Resize accordingly the synthesized image
            #         simg = cv2.resize(simg, (display_height, display_height), interpolation=cv2.INTER_CUBIC)
            #         h += display_height
            #
            #         # Concatenate and show the images
            #         img = np.concatenate((simg, img), axis=0)
            #         cv2.imshow('Visuorreactive Demo', img)
            #
            #         # display_width = int(4/3*display_height)
            #         # Resize input image from the camera
            #         # img = cv2.resize(image, (display_width, display_height))
            #         # w, h = display_width, display_height
            #         # Resize accordingly the synthesized image
            #         # simg = cv2.resize(simg, (display_height, display_height), interpolation=cv2.INTER_CUBIC)
            #         # w += h
            #         # img = np.concatenate((img, simg), axis=1)
            #         # cv2.imshow('Visuorreactive Demo', img)
            #     else:
            #         # Resize the synthesized image to the desired display height/width
            #         simg = cv2.resize(simg, (display_height, display_height))
            #         w, h = display_height, display_height
            #         cv2.imshow('Visuorreactive Demo - Only Synth Image', simg)
            #
            #     counter += 1
            #
            #     key = cv2.waitKey(1)
            #     # User presses 'ESC' to exit
            #     if key == 27:
            #         break
            #     elif key == 32:
            #     # Transition from not recording to recording
            #         if not recording_flag:
            #             print('Recording started')
            #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #             out = cv2.VideoWriter(f'visualreactive_input_v2_{model_type}.mp4', fourcc, fps, (w, h))
            #             recording_flag = True
            #         else:
            #             print('Recording stopped')
            #             recording_flag = False
            #             out.release()
            #
            #     if recording_flag:
            #         out.write(img)
            # cam.release()

    elif v3:
        # Set number of rows and columns for the generated "mirror"
        num_frames = 900
        nrows = 16
        ncols = 20
        shape = [num_frames, nrows * ncols, G.z_dim]
        # Generate a loop of images
        all_latents = np.random.RandomState(seed).randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[3.0 * 30, 0, 0], mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))
        all_latents = torch.from_numpy(all_latents).to(device)

        c = 0
        while cam.isOpened():
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            canny = cv2.Canny(blur, 10, 70)
            ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)

            # Truncation trick
            # Reshape the mask to the same size as the latent vector
            mask = cv2.resize(mask, (nrows, ncols), interpolation=cv2.INTER_AREA)
            mask = mask.astype(np.float32) / float(mask.max())
            trunc = torch.from_numpy(mask).view(-1, 1, 1).to(device)
            trunc = 1.0 - trunc

            # Get the latent vectors
            z = all_latents[c % num_frames]
            w = G.mapping(z, None)
            w = w * trunc + G.mapping.w_avg * (1 - trunc)

            c += 1

            simg = gen_utils.w_to_img(G, w, truncation_psi=truncation_psi)
            simg = gen_utils.create_image_grid(simg, (ncols, nrows))
            simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)

            # Resize the synthesized image to the desired display height/width
            simg = cv2.resize(simg, (int(display_height * ncols / nrows), display_height))

            cv2.imshow('Video feed', simg)

            counter += 1
            # FPS counter
            if (time.time() - start_time) > x and verbose:
                print(f"FPS: {counter / (time.time() - start_time):0.2f}")
                counter = 0
                start_time = time.time()

            key = cv2.waitKey(1)
            # User presses 'ESC' to exit
            if key == 27:
                break
        cam.release()


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    live_visual_reactive()


# ----------------------------------------------------------------------------

import scipy
import numpy as np
from PIL import Image
def recreate_aligned_images(img, eyes, output_size=512, transform_size=2048, enable_padding=False, image_type="raw"):
    eye_left, eye_right = eyes
    eye_left = eye_left.astype('float')
    eye_right = eye_right.astype('float')
    eye_to_eye   = eye_right - eye_left
    eye_avg      = (eye_left + eye_right) * 0.5
    
    rescale_ratio = 1.8
    x = eye_to_eye.copy()
    x /= np.hypot(*x)
    x *= np.hypot(*eye_to_eye) * rescale_ratio
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg
    
    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2
    
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        if image_type == "segmentation":
            img = img.resize(rsize, Image.NEAREST)
        else:
            img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
        
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
        
    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    enable_padding = enable_padding and (image_type == "raw")
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    if image_type == "segmentation":
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.NEAREST)
    else:
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        if image_type == "segmentation":
            img = img.resize((output_size, output_size), Image.NEAREST)
        else:
            img = img.resize((output_size, output_size), Image.ANTIALIAS)
    return img
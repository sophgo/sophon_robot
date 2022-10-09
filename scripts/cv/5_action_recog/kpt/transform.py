import numpy as np
import cv2

########################
### postprocess utils
########################
def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords

def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
     # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = (preds[..., 0] + preds[..., 1] * W).flatten()
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5
    
    # Transform back to the image
    #import pdb;pdb.set_trace()
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center, scale, [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

########################
### preprocess utils
########################
class TopDownGetBboxCenterScale:
    """Convert bbox from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required key: 'bbox', 'ann_info'

    Modifies key: 'center', 'scale'

    Args:
        padding (float): bbox padding scale that will be multilied to scale.
            Default: 1.25
    """
    # Pixel std is 200.0, which serves as the normalization factor to
    # to calculate bbox scales.
    pixel_std: float = 200.0

    def __init__(self, padding: float = 1.25):
        self.padding = padding

    def __call__(self, results):

        if 'center' in results and 'scale' in results:
            warnings.warn(
                'Use the "center" and "scale" that already exist in the data '
                'sample. The padding will still be applied.')
            results['scale'] *= self.padding
        else:
            bbox = results['bbox']
            image_size = results['ann_info']['image_size']
            aspect_ratio = image_size[0] / image_size[1]

            center, scale = bbox_xywh2cs(
                bbox,
                aspect_ratio=aspect_ratio,
                padding=self.padding,
                pixel_std=self.pixel_std)

            results['center'] = center
            results['scale'] = scale
        return results

class TopDownAffine:
    """Affine transform the image to make input.

    Required key:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified key:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        #joints_3d = results['joints_3d']
        #joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]

            # joints_3d[:, 0:2] = \
            #    warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
            # for i in range(results['ann_info']['num_joints']):
            #     if joints_3d_visible[i, 0] > 0.0:
            #         joints_3d[i,
            #                   0:2] = affine_transform(joints_3d[i, 0:2], trans)

        results['img'] = img
        # results['joints_3d'] = joints_3d
        # results['joints_3d_visible'] = joints_3d_visible

        return results

def bbox_xywh2cs(bbox, aspect_ratio, padding=1., pixel_std=200.):
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Single bbox in (x, y, w, h)
        aspect_ratio (float): The expected bbox aspect ratio (w over h)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.0
        pixel_std (float): The scale normalization factor. Default: 200.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = bbox[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w, h], dtype=np.float32) / pixel_std
    scale = scale * padding

    return center, scale

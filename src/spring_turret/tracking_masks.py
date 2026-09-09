"""Geometry and display helpers for existing full-resolution CPU masks.

Centroids use original binary masks before overlap composition or display resize.
No model calls, GPU work, thresholds, boxes, or temporal state are changed.
One indexed PNG avoids sending a separate full-frame image for every instance.
"""
import base64
import colorsys
import io
import math

MAX_PNG_BYTES = 384 * 1024  # Base64 + boxes stay below the 1 MiB IPC metadata cap.


def valid_mask_centroid(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in value))


def mask_centroid(mask):
    """Mean foreground pixel center as normalized [x,y], or None if empty.

    Marginal counts avoid allocating coordinate arrays for every foreground
    pixel. Pixel areas span [0,width] x [0,height], with centers at index+.5.
    """
    import numpy as np
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_ or min(mask.shape) < 1:
        raise ValueError('Centroid requires a nonzero-size 2D binary mask')
    height, width = mask.shape
    columns = np.count_nonzero(mask, axis=0)
    area = int(columns.sum())
    if area == 0:
        return None
    rows = np.count_nonzero(mask, axis=1)
    return [float((columns @ np.arange(width) / area + .5) / width),
            float((rows @ np.arange(height) / area + .5) / height)]


def instance_color(base, instance_id):
    """Stable nearby hue/lightness, independent of frame order or visibility."""
    rgb = tuple(int(base[i:i+2], 16)/255 for i in (1,3,5))
    hue, _, _ = colorsys.rgb_to_hls(*rgb)
    # Adjacent IDs spread over the class's +/-12 degree hue band. Lightness
    # makes neighboring instances distinguishable without changing color family.
    offset = (int(instance_id) * .618033988749895) % 1
    lightness = .43 + .25 * ((int(instance_id) * .754877666246693) % 1)
    saturation = .65 + .25 * ((int(instance_id) * .414213562373095) % 1)
    color = colorsys.hls_to_rgb((hue + (offset-.5)/15) % 1, lightness, saturation)
    return '#' + ''.join(f'{round(c*255):02x}' for c in color)


def encode_mask_overlay(instances, width, height):
    """instances = (binary mask, RGB hex color, stable ID) for visible objects.

    Prefer original camera resolution. Only pathological, incompressible display
    masks are reduced to keep UI publication bounded; inference/servo geometry
    always retain the original masks and boxes.
    """
    if not instances:
        return None
    import numpy as np
    from PIL import Image
    if not 0 < width <= 8192 or not 0 < height <= 8192 or len(instances) > 128:
        raise ValueError('Invalid display mask dimensions or instance count')
    labels = np.zeros((height, width), dtype=np.uint8)
    palette = [0] * 768
    # Large masks first; smaller overlapping objects remain visible. Stable IDs
    # break ties, so order changes do not cause overlap/color flicker.
    ordered = sorted(instances, key=lambda item: (-np.count_nonzero(item[0]), item[2]))
    for index, (mask, color, _) in enumerate(ordered, 1):
        if mask.shape != labels.shape or mask.dtype != np.bool_:
            raise ValueError('Tracking mask must match the original camera frame')
        labels[mask] = index
        palette[index*3:index*3+3] = [int(color[i:i+2],16) for i in (1,3,5)]
    image = Image.fromarray(labels).convert('P')
    image.putpalette(palette)
    alpha = bytes([0] + [112]*len(ordered) + [0]*(255-len(ordered)))
    needed = math.ceil(math.log2(len(ordered)+1))
    bits = next(b for b in (1,2,4,8) if b >= needed)
    while True:
        output = io.BytesIO()
        image.save(output, format='PNG', bits=bits, transparency=alpha, compress_level=1)
        data = output.getvalue()
        if len(data) <= MAX_PNG_BYTES:
            break
        image = image.resize((max(1,image.width//2), max(1,image.height//2)), Image.Resampling.NEAREST)
    return {'format':'indexed-png', 'png':base64.b64encode(data).decode('ascii'),
            'width':image.width, 'height':image.height,
            'source_width':width, 'source_height':height}

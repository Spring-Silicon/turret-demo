"""Display-only, lossless palette overlays from existing CPU tracking masks.

No model calls, GPU work, thresholds, boxes, or temporal state are changed.
One indexed PNG avoids sending a separate full-frame image for every instance.
"""
import base64
import colorsys
import io
import math

MAX_PNG_BYTES = 384 * 1024  # Base64 + boxes stay below the 1 MiB IPC metadata cap.


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

import numpy as np

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def rgb_to_hsv(r, g, b):
    """
    Converts RGB (0-255) to HSV where H is [0, 360], S is [0, 1], V is [0, 1].
    Matches the JS implementation in canvas-utils.js exactly.
    """
    rr = r / 255.0
    gg = g / 255.0
    bb = b / 255.0
    max_c = max(rr, gg, bb)
    min_c = min(rr, gg, bb)
    d = max_c - min_c
    
    h = 0.0
    if d != 0:
        if max_c == rr:
            h = ((gg - bb) / d + (6.0 if gg < bb else 0.0)) * 60.0
        elif max_c == gg:
            h = ((bb - rr) / d + 2.0) * 60.0
        else:
            h = ((rr - gg) / d + 4.0) * 60.0
            
    s = 0.0 if max_c == 0.0 else d / max_c
    return h, s, max_c

def sample_hsv(img_rgb, x, y, radius):
    """
    Samples pixels in a circle of radius around (x, y) on the img_rgb (HWC NumPy array),
    converting each to our custom rgb_to_hsv format and averaging the results.
    """
    h, w, _ = img_rgb.shape
    points = []
    r = max(1, int(radius))
    
    for oy in range(-r, r + 1, 2):
        for ox in range(-r, r + 1, 2):
            if ox*ox + oy*oy > r*r:
                continue
            px = max(0, min(w - 1, int(round(x + ox))))
            py = max(0, min(h - 1, int(round(y + oy))))
            
            # opencv stores as BGR, so if img_rgb is BGR or RGB, let's look at color
            # We assume input to sample_hsv is RGB
            color = img_rgb[py, px]
            hsv = rgb_to_hsv(color[0], color[1], color[2])
            points.append(hsv)
            
    if not points:
        return 0.0, 0.0, 0.0
        
    sum_h = sum(p[0] for p in points)
    sum_s = sum(p[1] for p in points)
    sum_v = sum(p[2] for p in points)
    n = len(points)
    return sum_h / n, sum_s / n, sum_v / n

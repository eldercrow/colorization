# Joint Bilateral Upsampling of the result color image w.r.t the input gray image.
# http://johanneskopf.de/publications/jbu/
#
import cv2
import numpy as np

def jbupsample(img, g_img, sigma_s=2.0, sigma_r=0.05):
    # perform joint bilateral upsample
    # upsame img to the size of g_img
    #
    hh_g = g_img.shape[0]
    ww_g = g_img.shape[1]

    step_y = float(hh_g) / float(img.shape[0])
    step_x = float(ww_g) / float(img.shape[1])

    I = cv2.resize(img, (ww_g, hh_g))
    G = g_img

    if I.dtype == 'uint8':
        I = I.astype('float32') / 255.0
    if G.dtype == 'uint8':
        G = G.astype('float32') / 255.0

    I = np.reshape(I, (hh_g, ww_g, -1))
    G = np.reshape(G, (hh_g, ww_g, -1))

    fr = np.round(sigma_s * 3.0)
    s_range = np.arange(-fr, fr+1.0, 1.0)
    offset_x = s_range * step_x
    offset_y = s_range * step_y

    offset_x = np.round(offset_x).astype(int)
    offset_y = np.round(offset_y).astype(int)

    w_spatial = np.exp(-0.5 * s_range*s_range/sigma_s/sigma_s)

    r_img = np.zeros_like(I)
    w_sum = np.zeros((hh_g, ww_g, 1))

    for (iy, oy) in enumerate(offset_y):
        y0_c = max(oy, 0)
        y0_n = max(-oy, 0)
        y1_c = min(hh_g+oy, hh_g)
        y1_n = min(hh_g-oy, hh_g)

        for (ix, ox) in enumerate(offset_x):
            x0_c = max(ox, 0)
            x0_n = max(-ox, 0)
            x1_c = min(ww_g+ox, ww_g)
            x1_n = min(ww_g-ox, ww_g)

            w_s = w_spatial[iy] * w_spatial[ix]
            dg = G[y0_c:y1_c, x0_c:x1_c, :] - G[y0_n:y1_n, x0_n:x1_n, :]
            dg = np.sum(dg*dg, axis=2, keepdims=True)
            w_r = np.exp(-0.5 * dg/sigma_r/sigma_r)

            r_img[y0_c:y1_c, x0_c:x1_c, :] += I[y0_n:y1_n, x0_n:x1_n, :] * w_r * w_s
            w_sum[y0_c:y1_c, x0_c:x1_c, :] += w_r * w_s

    r_img /= w_sum

    return r_img

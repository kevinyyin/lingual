import base64, sys, cv2, numpy as np
sys.path.insert(0, 'spike2')
from v_shots import page, dec, grid
from playwright.sync_api import sync_playwright
V = [("x+", [[0.8,0,0],[0,0,0],[0,0,0]]), ("x-", [[-0.8,0,0],[0,0,0],[0,0,0]]), ("y+", [[0,0.8,0],[0,0,0],[0,0,0]]), ("y-", [[0,-0.8,0],[0,0,0],[0,0,0]]),
     ("z+", [[0,0,0.8],[0,0,0],[0,0,0]]), ("z-", [[0,0,-0.8],[0,0,0],[0,0,0]]), ("prox y+ dist y+", [[0,0,0],[0,0.8,0],[0,0.8,0]]), ("prox z+ dist z+", [[0,0,0],[0,0,0.8],[0,0,0.8]])]
with sync_playwright() as p:
    b, pg = page(p, "handcam=1", 640, 640)
    ims = [dec(pg.evaluate(f"window.thumbTest({e})")) for _, e in V]
    cv2.imwrite(sys.argv[1], grid(ims, [n for n, _ in V], 4)); b.close()

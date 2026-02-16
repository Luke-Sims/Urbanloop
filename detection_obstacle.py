import cv2
import numpy as np
from collections import Counter

# Détecte les pixels plus clairs que la majorité sur une ligne
def detect_outliers_lighter_than_majority(frame, threshold=20, method='median'):
    height, width = frame.shape
    anomaly_mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        line = frame[y, :]
        if method == 'median':
            reference = np.median(line)
        elif method == 'mean':
            reference = np.mean(line)
        elif method == 'mode':
            counts = Counter(line)
            reference = counts.most_common(1)[0][0]
        else:
            reference = np.median(line)

        diff = frame[y, :].astype(np.int16) - reference
        anomaly_mask[y, diff > threshold] = 255

    return anomaly_mask

# Regroupe les pixels détectés en objets.
def group_anomalies(anomaly_mask, min_area=50):
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
    dilated_mask = cv2.dilate(closed_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def nothing(x):
    pass

def process_video_realtime(depth_path, rgb_path):
    cap_depth = cv2.VideoCapture(depth_path)
    cap_rgb = cv2.VideoCapture(rgb_path)

    if not cap_depth.isOpened() or not cap_rgb.isOpened():
        print("Erreur: Impossible d'ouvrir les vidéos.")
        return

    dw = cap_depth.get(cv2.CAP_PROP_FRAME_WIDTH)
    dh = cap_depth.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rw = cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)
    rh = cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)

    scale_x_base = rw / dw
    scale_y_base = rh / dh

    win_rgb = "Calibration RGB (Sortie)"
    win_depth = "Detection Source (Depth)"
    cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    # Réglages des curseurs
    cv2.createTrackbar("Seuil", win_rgb, 12, 100, nothing)
    cv2.createTrackbar("Aire Min", win_rgb, 5000, 20000, nothing)
    cv2.createTrackbar("Lissage", win_rgb, 50, 99, nothing) # 0 = nerveux, 99 = très fluide
    cv2.createTrackbar("Decalage X", win_rgb, 170, 400, nothing)
    cv2.createTrackbar("Decalage Y", win_rgb, 5, 400, nothing)
    cv2.createTrackbar("Zoom X", win_rgb, 120, 300, nothing)
    cv2.createTrackbar("Zoom Y", win_rgb, 135, 300, nothing)

    # --- MÉMOIRE POUR LE LISSAGE ---
    # On stocke les rectangles sous forme de liste de tuples (x, y, w, h)
    prev_rects = []
    
    paused = False
    flip_image = True
    current_method = 'median'
    methods = ['median', 'mean', 'mode']
    cur_frame_d, cur_frame_r = None, None

    while True:
        t_val = cv2.getTrackbarPos("Seuil", win_rgb)
        area_val = cv2.getTrackbarPos("Aire Min", win_rgb)
        # alpha : poids de l'ancienne position (0.0 à 0.99)
        alpha = cv2.getTrackbarPos("Lissage", win_rgb) / 100.0
        off_x = cv2.getTrackbarPos("Decalage X", win_rgb) - 100
        off_y = cv2.getTrackbarPos("Decalage Y", win_rgb) - 100
        zx = cv2.getTrackbarPos("Zoom X", win_rgb) / 100.0
        zy = cv2.getTrackbarPos("Zoom Y", win_rgb) / 100.0

        if not paused:
            ret_d, frame_d = cap_depth.read()
            ret_r, frame_r = cap_rgb.read()
            if not ret_d or not ret_r: break

            if flip_image:
                cur_frame_d = cv2.flip(frame_d, 0)
                cur_frame_r = cv2.flip(frame_r, 0)
            else:
                cur_frame_d, cur_frame_r = frame_d, frame_r

        if cur_frame_d is not None:
            gray_d = cv2.cvtColor(cur_frame_d, cv2.COLOR_BGR2GRAY)
            depth_output = cv2.cvtColor(gray_d, cv2.COLOR_GRAY2BGR)
            rgb_output = cur_frame_r.copy()

            mask = detect_outliers_lighter_than_majority(gray_d, t_val, current_method)
            contours = group_anomalies(mask, area_val)
            
            current_detected_rects = [cv2.boundingRect(cnt) for cnt in contours]
            new_rects_to_store = []

            for (xr, yr, wr, hr) in current_detected_rects:
                # --- LOGIQUE DE LISSAGE ---
                # On cherche si un rectangle existait déjà à la frame précédente près d'ici
                best_match = None
                min_dist = 50 # Distance max en pixels pour considérer que c'est le même objet

                for i, (px, py, pw, ph) in enumerate(prev_rects):
                    dist = np.sqrt((xr - px)**2 + (yr - py)**2)
                    if dist < min_dist:
                        best_match = (px, py, pw, ph)
                        break

                if best_match is not None and not paused:
                    # Formule d'interpolation : (Ancien * alpha) + (Nouveau * (1 - alpha))
                    nx = int(best_match[0] * alpha + xr * (1 - alpha))
                    ny = int(best_match[1] * alpha + yr * (1 - alpha))
                    nw = int(best_match[2] * alpha + wr * (1 - alpha))
                    nh = int(best_match[3] * alpha + hr * (1 - alpha))
                else:
                    # Premier enregistrement ou objet trop loin : pas de lissage
                    nx, ny, nw, nh = xr, yr, wr, hr

                new_rects_to_store.append((nx, ny, nw, nh))

                # --- AFFICHAGE ET PROJECTION ---
                # Couleur selon profondeur
                roi = gray_d[ny:ny+nh, nx:nx+nw]
                color = (0, 255, 0)
                if roi.size > 0:
                    ratio = np.clip(np.mean(roi) / 30.0, 0, 1)
                    color = (0, int(255*(1-ratio)), int(255*ratio))

                cv2.rectangle(depth_output, (nx, ny), (nx + nw, ny + nh), color, 2)

                # Projection RGB
                w_rgb = int(nw * scale_x_base * zx)
                h_rgb = int(nh * scale_y_base * zy)
                cx_rgb = (nx + nw/2) * scale_x_base
                cy_rgb = (ny + nh/2) * scale_y_base
                x_rgb = int(cx_rgb - w_rgb/2) + off_x
                y_rgb = int(cy_rgb - h_rgb/2) + off_y

                cv2.rectangle(rgb_output, (x_rgb, y_rgb), (x_rgb + w_rgb, y_rgb + h_rgb), color, 3)

            # Mise à jour de la mémoire
            if not paused:
                prev_rects = new_rects_to_store

            cv2.imshow(win_rgb, rgb_output)
            cv2.imshow(win_depth, depth_output)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('f'): flip_image = not flip_image
        elif key == ord('m'):
            current_method = methods[(methods.index(current_method)+1) % len(methods)]

    cap_depth.release()
    cap_rgb.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_realtime("personnes_et_objets_en_mouvement/videoDepth.avi","personnes_et_objets_en_mouvement/videoRgb.avi")

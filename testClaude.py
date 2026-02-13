import cv2
import numpy as np
import math

# ----------------------------
# Helpers
# ----------------------------
def ema(prev, new, alpha=0.3):
    """Exponential Moving Average pour lisser les détections entre frames"""
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


def fit_polynomial_weighted(segments, degree=2):
    """
    Ajustement polynomial pondéré par la longueur des segments
    Permet de gérer les rails courbes
    """
    if not segments:
        return None
    
    xs, ys, ws = [], [], []
    for x1, y1, x2, y2, weight in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        ws.extend([weight, weight])
    
    if len(xs) < 3:
        return None
    
    try:
        # Ajustement polynomial pondéré
        poly = np.polyfit(ys, xs, degree, w=ws)
        return poly
    except:
        return None


def draw_polynomial_line(img, poly, y_start, y_end, color=(0, 255, 0), thickness=3):
    """Dessine une ligne polynomiale sur l'image"""
    if poly is None:
        return
    
    y_points = np.linspace(y_start, y_end, 50)
    x_points = np.polyval(poly, y_points)
    
    # Créer les points pour polylines
    points = np.array([np.vstack([x_points, y_points]).T], dtype=np.int32)
    cv2.polylines(img, points, False, color, thickness)


def calculate_vanishing_point(left_poly, right_poly, y_top):
    """Calcule le point de fuite à partir des deux polynômes"""
    if left_poly is None or right_poly is None:
        return None
    
    x_left = np.polyval(left_poly, y_top)
    x_right = np.polyval(right_poly, y_top)
    
    vp_x = (x_left + x_right) / 2
    return int(vp_x), int(y_top)


# ----------------------------
# Détection principale
# ----------------------------
def detect_rails(frame_bgr, 
                 prev_left_poly=None, 
                 prev_right_poly=None,
                 debug=False,
                 canny_low=50, 
                 canny_high=150,
                 roi_top_ratio=0.60,
                 min_len=40,
                 max_gap=30,
                 slope_min=0.35,
                 split_x_ratio=0.5):
    """
    Détection robuste de rails avec:
    - Ajustement polynomial pour rails courbes
    - Lissage temporel avec EMA
    - Contraintes géométriques (point de fuite, zone de convergence)
    """
    
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    
    # 1. Définir la ROI (partie basse de l'image)
    y_roi_start = int(h * roi_top_ratio)
    roi = frame_bgr[y_roi_start:h, :]
    
    # 2. Prétraitement
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Réduction du bruit
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 3. Détection des contours
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)
    
    # Morphologie pour nettoyer
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # 4. Détection de lignes avec Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=60,
        minLineLength=min_len,
        maxLineGap=max_gap
    )
    
    # 5. Définir les contraintes géométriques
    cx_center = w * 0.5
    y_top = y_roi_start
    y_bottom = h - 1
    
    # Zone de convergence (point de fuite)
    vp_min = w * 0.45
    vp_max = w * 0.55
    max_vp_tolerance = w * 0.25
    
    # Zone de passage au bas de l'image
    bottom_min = w * 0.35
    bottom_max = w * 0.65
    
    # Pente minimale pour filtrer les lignes horizontales
    min_abs_slope = slope_min
    
    # Ligne de séparation gauche/droite
    split_x = w * split_x_ratio
    
    # 6. Filtrer et classer les segments
    candidates = []
    
    if lines is not None:
        for line in lines:
            x1, y1_rel, x2, y2_rel = line[0]
            
            # Convertir les coordonnées relatives à la ROI en coordonnées absolues
            y1 = y1_rel + y_roi_start
            y2 = y2_rel + y_roi_start
            
            # Calculer la longueur
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_len:
                continue
            
            # Calculer la pente
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:
                continue
            
            slope = dy / dx
            
            # Filtrer les pentes trop faibles (lignes quasi-horizontales)
            if abs(slope) < min_abs_slope:
                continue
            
            # Extrapoler vers le haut et le bas
            # Vers le haut (point de fuite)
            if dy != 0:
                m = dx / dy
                b_x = x1 - m * y1
                x_top = m * y_top + b_x
                x_bot = m * y_bottom + b_x
            else:
                continue
            
            # Vérifier la convergence vers le point de fuite
            if not (vp_min - max_vp_tolerance <= x_top <= vp_max + max_vp_tolerance):
                continue
            
            # Vérifier le passage en bas de l'image
            if not (bottom_min <= x_bot <= bottom_max):
                continue
            
            candidates.append((x1, y1, x2, y2, length, x_bot, slope))
    
    # 7. Séparer gauche/droite selon la position au bas de l'image
    left_segments = []
    right_segments = []
    
    for x1, y1, x2, y2, length, x_bot, slope in candidates:
        if x_bot < split_x:
            left_segments.append((x1, y1, x2, y2, length))
        else:
            right_segments.append((x1, y1, x2, y2, length))
    
    # 8. Garder les meilleurs segments (tri par longueur)
    left_segments = sorted(left_segments, key=lambda s: s[4], reverse=True)[:3]
    right_segments = sorted(right_segments, key=lambda s: s[4], reverse=True)[:3]
    
    # 9. Ajustement polynomial pour chaque côté
    left_poly = fit_polynomial_weighted(left_segments, degree=2)
    right_poly = fit_polynomial_weighted(right_segments, degree=2)
    
    # 10. Lissage temporel avec EMA
    if prev_left_poly is not None and left_poly is not None:
        left_poly = ema(prev_left_poly, left_poly, alpha=0.25)
    
    if prev_right_poly is not None and right_poly is not None:
        right_poly = ema(prev_right_poly, right_poly, alpha=0.25)
    
    # 11. Dessiner les résultats
    if debug:
        # ROI
        cv2.rectangle(overlay, (0, y_roi_start), (w-1, h-1), (100, 100, 100), 2)
        
        # Zones de contraintes
        cv2.line(overlay, (int(vp_min), y_top), (int(vp_min), y_top + 30), (255, 255, 255), 2)
        cv2.line(overlay, (int(vp_max), y_top), (int(vp_max), y_top + 30), (255, 255, 255), 2)
        
        cv2.line(overlay, (int(bottom_min), y_bottom), (int(bottom_min), y_bottom - 40), (255, 255, 255), 2)
        cv2.line(overlay, (int(bottom_max), y_bottom), (int(bottom_max), y_bottom - 40), (255, 255, 255), 2)
        
        cv2.line(overlay, (int(split_x), y_bottom), (int(split_x), y_bottom - 80), (255, 255, 0), 2)
        
        # Segments détectés
        for x1, y1, x2, y2, _ in left_segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 100, 100), 2)
        
        for x1, y1, x2, y2, _ in right_segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (100, 100, 255), 2)
        
        # Statistiques
        nb_lines = 0 if lines is None else len(lines)
        cv2.putText(overlay, 
                   f"Lines={nb_lines} Cand={len(candidates)} L={len(left_segments)} R={len(right_segments)}",
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 12. Dessiner les rails finaux (courbes polynomiales)
    if left_poly is not None:
        draw_polynomial_line(overlay, left_poly, y_bottom, y_top, color=(0, 255, 0), thickness=4)
    
    if right_poly is not None:
        draw_polynomial_line(overlay, right_poly, y_bottom, y_top, color=(0, 255, 0), thickness=4)
    
    # Calculer et afficher le point de fuite
    if left_poly is not None and right_poly is not None:
        vp = calculate_vanishing_point(left_poly, right_poly, y_top)
        if vp is not None:
            cv2.circle(overlay, vp, 8, (0, 255, 255), -1)
            cv2.circle(overlay, vp, 12, (0, 255, 255), 2)
    
    return overlay, edges, left_poly, right_poly


def main():
    """Fonction principale pour le traitement vidéo"""
    
    # Configuration
    source = "/mnt/user-data/uploads/1770976285180_image.png"  # Remplacer par votre vidéo
    force_rotate_180 = False  # Mettre à True si nécessaire
    
    # Vérifier si c'est une vidéo ou une image
    if source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(source)
    else:
        # Pour tester sur l'image
        img = cv2.imread(source)
        if img is None:
            print(f"Impossible de charger: {source}")
            return
        
        if force_rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        
        # Traiter l'image
        overlay, edges, _, _ = detect_rails(img, debug=True, split_x_ratio=0.5)
        
        # Sauvegarder et afficher
        cv2.imwrite('/home/claude/rails_detected.png', overlay)
        cv2.imwrite('/home/claude/edges_detected.png', edges)
        
        print("✅ Détection terminée!")
        print("📁 Fichiers sauvegardés:")
        print("   - /home/claude/rails_detected.png")
        print("   - /home/claude/edges_detected.png")
        return
    
    # Traitement vidéo
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {source}")
    
    # Variables pour le lissage temporel
    prev_left_poly = None
    prev_right_poly = None
    frame_count = 0
    
    print("🎥 Traitement vidéo en cours...")
    print("Appuyez sur 'q' ou 'ESC' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if force_rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Détection avec lissage temporel
        overlay, edges, left_poly, right_poly = detect_rails(
            frame,
            prev_left_poly=prev_left_poly,
            prev_right_poly=right_poly,
            debug=True,
            split_x_ratio=0.50  # Ajuster selon votre vidéo
        )
        
        # Mise à jour pour la prochaine frame
        if left_poly is not None:
            prev_left_poly = left_poly
        if right_poly is not None:
            prev_right_poly = right_poly
        
        # Affichage
        cv2.imshow("Rails Detection", overlay)
        cv2.imshow("Edges (ROI)", edges)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count} traitée...")
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC ou 'q'
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Traitement terminé! {frame_count} frames traitées")


if __name__ == "__main__":
    main()
import cv2
import numpy as np

# ----------------------------
# Paramètres de Stabilisation
# ----------------------------
class LaneStabilizer:
    def __init__(self, alpha=0.2):
        self.alpha = alpha  # Facteur de lissage (0.1 = très stable, 0.9 = très réactif)
        self.coeffs = None

    def smooth(self, new_coeffs):
        if new_coeffs is None:
            return self.coeffs
        if self.coeffs is None:
            self.coeffs = new_coeffs
        else:
            self.coeffs = (1 - self.alpha) * self.coeffs + self.alpha * new_coeffs
        return self.coeffs

# ----------------------------
# Fonctions Utiles
# ----------------------------
def fit_poly(segments):
    """Calcule un polynôme de degré 2 (x = ay^2 + by + c) à partir de segments."""
    if not segments or len(segments) < 2:
        return None
    
    pts_x = []
    pts_y = []
    for x1, y1, x2, y2, _ in segments:
        pts_x.extend([x1, x2])
        pts_y.extend([y1, y2])
    
    try:
        # On utilise un degré 2 pour capturer la courbure
        coeffs = np.polyfit(pts_y, pts_x, 2)
        return coeffs
    except np.RankWarning:
        return None

def draw_rail_poly(img, coeffs, y_start, y_end, color, thickness=5):
    """Génère les points de la courbe et les dessine."""
    if coeffs is None:
        return
    
    # Générer des points Y pour la courbe
    plot_y = np.linspace(y_start, y_end - 1, 20)
    # Calculer X = ay^2 + by + c
    plot_x = coeffs[0] * plot_y**2 + coeffs[1] * plot_y + coeffs[2]
    
    # Mise en forme pour cv2.polylines
    pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], np.int32)
    cv2.polylines(img, pts, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

# ----------------------------
# Détection Principale
# ----------------------------
def process_frame(frame, stab_left, stab_right):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # 1. Région d'intérêt (ROI) - On prend le bas de l'image (60%)
    y_roi_top = int(h * 0.6)
    roi = frame[y_roi_top:h, :]
    
    # 2. Prétraitement pour isoler les rails
    # Passage en HLS pour filtrer le ballast (souvent gris/peu saturé)
    hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    _, s_channel = cv2.threshold(hls[:, :, 2], 50, 255, cv2.THRESH_BINARY)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Combinaison du gris et du masque de saturation pour nettoyer l'image
    refined_gray = cv2.bitwise_and(blurred, s_channel)
    edges = cv2.Canny(refined_gray, 50, 150)
    
    # 3. Détection de lignes (Hough)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=70)
    
    left_segments, right_segments = [], []
    mid_x = w / 2

    if lines is not None:
        for line in lines:
            x1, y1r, x2, y2r = line[0]
            # Coordonnées globales
            y1, y2 = y1r + y_roi_top, y2r + y_roi_top
            
            # Calcul de la pente
            if x2 - x1 == 0: continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Filtrage : On ignore les lignes trop horizontales (pente < 0.3)
            if abs(slope) < 0.35: continue
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Séparation Gauche/Droite selon le milieu de l'image
            if x1 < mid_x and x2 < mid_x:
                left_segments.append((x1, y1, x2, y2, length))
            elif x1 > mid_x and x2 > mid_x:
                right_segments.append((x1, y1, x2, y2, length))

    # 4. Fit Polynomial + Lissage Temporel
    raw_left_coeffs = fit_poly(left_segments)
    raw_right_coeffs = fit_poly(right_segments)
    
    smooth_left = stab_left.smooth(raw_left_coeffs)
    smooth_right = stab_right.smooth(raw_right_coeffs)

    # 5. Dessin des courbes
    # Rail Gauche (Bleu)
    draw_rail_poly(overlay, smooth_left, y_roi_top, h, (255, 0, 0), 6)
    # Rail Droit (Rouge)
    draw_rail_poly(overlay, smooth_right, y_roi_top, h, (0, 0, 255), 6)
    
    # Optionnel : Remplir l'espace entre les rails (zone de voie)
    if smooth_left is not None and smooth_right is not None:
        fill_lane(overlay, smooth_left, smooth_right, y_roi_top, h)

    return overlay, edges

def fill_lane(img, left_coeffs, right_coeffs, y_start, y_end):
    """Dessine un polygone translucide entre les deux rails."""
    plot_y = np.linspace(y_start, y_end - 1, 15)
    left_x = left_coeffs[0] * plot_y**2 + left_coeffs[1] * plot_y + left_coeffs[2]
    right_x = right_coeffs[0] * plot_y**2 + right_coeffs[1] * plot_y + right_coeffs[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))
    
    lane_mask = np.zeros_like(img)
    cv2.fillPoly(lane_mask, np.int_([pts]), (0, 255, 0))
    cv2.addWeighted(img, 1.0, lane_mask, 0.3, 0, img)

# ----------------------------
# Boucle Principale
# ----------------------------
def main():
    cap = cv2.VideoCapture("videoRgb.mp4") # Remplace par 0 pour webcam
    
    # Initialisation des stabilisateurs
    stab_left = LaneStabilizer(alpha=0.15)
    stab_right = LaneStabilizer(alpha=0.15)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Correction d'orientation (si nécessaire)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        result, debug_edges = process_frame(frame, stab_left, stab_right)
        
        cv2.imshow("Detection Rails Courbes", result)
        cv2.imshow("Edges", debug_edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# Urbanloop

Projet de véhicule autonome utilisant la vision par ordinateur pour la détection des rails et des obstacles, avec communication sans fil IrDA.

## Présentation
Retrouvez la présentation complète du projet ici : [Lien vers la présentation](https://docs.google.com/presentation/d/1uMGySIxdCz8Myz7OVVhOK5eT4uFbThtZT8y8aCtLFhE/edit?usp=sharing)

## Structure du projet

```
Urbanloop/
├── rail.py                  # Détection des rails par transformée de Hough
├── rail_v2.py               # Détection alternative par traçage de contours
├── detection_obstacle.py    # Détection d'obstacles via vidéo profondeur/RGB
├── ESP's communications/
│   ├── prog_emetteur/       # Émetteur IrDA ESP32
│   └── prog_recepteur/       # Récepteur IrDA ESP32
└── vidéos/                  # Vidéos de test
```

## Prérequis

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Arduino/ESP32 toolchain (pour les programmes ESP)

Installer les dépendances Python :
```bash
pip install opencv-python numpy
```

## Modules

### Détection des rails (`rail.py`)

Détection des rails dans les images vidéo par détection de contours Canny et transformée de Hough. Retourne les positions des rails gauche et droit.

**Utilisation :**
```python
overlay, edges = detect_rails(frame_bgr, debug=True)
```

### Détection des rails v2 (`rail_v2.py`)

Méthode alternative de détection des rails par traçage des contours depuis le bas de l'image vers le haut.

**Utilisation :**
```python
overlay, edges, split_x_ratio = detect_rails(frame, debug=True, split_x_ratio=0.5)
```

### Détection d'obstacles (`detection_obstacle.py`)

Détection d'obstacles dans une vidéo de profondeur en cherchant les pixels plus clairs que la médiane de chaque ligne, puis regroupement en objets.

**Fonctionnalités :**
- Traitement temps réel avec synchronisation vidéo profondeur/RGB
- Paramètres ajustables via curseurs (seuil, aire min, lissage)
- Suivi d'objets avec interpolation pour une détection stable

**Utilisation :**
```python
process_video_realtime("depth.avi", "rgb.avi")
```

### Communication IrDA

Communication infrarouge basée sur ESP32 avec encodeur/décodeur MCP2120 :

- **Émetteur** (`prog_emetteur/`) : Envoie des commandes de vitesse (0-255) via IrDA
- **Récepteur** (`prog_recepteur/`) : Reçoit les commandes et contrôle le moteur/LEDs

## Matériel

- Capteur IR : MCP2120
- Microcontrôleur : ESP32
- Caméra : Vidéo profondeur + RGB

## Contrôles (Détection d'obstacles)

- `Espace` - Pause/reprise
- `f` - Inverser l'image
- `m` - Changer la méthode de détection (médiane/moyenne/mode)
- `q` - Quitter

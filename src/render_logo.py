"""
Render high-resolution 1024x1024 PNG logo for MaternaTriage using Matplotlib & Pillow.
Avoids cairosvg dependencies on Windows.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle, FancyBboxPatch
from matplotlib.path import Path
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
OUTPUT_PNG_PATH = os.path.join(ASSETS_DIR, "logo.png")


def render_logo(output_path: str = OUTPUT_PNG_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 10.24 x 10.24 inches at 100 DPI = exactly 1024 x 1024 pixels
    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=100)
    fig.patch.set_facecolor("#0F172A") # Deep slate backdrop for contrast
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.axis("off")

    # Card background (rounded canvas)
    card = FancyBboxPatch(
        (40, 40), 920, 920,
        boxstyle="round,pad=0,rounding_size=80",
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        linewidth=4,
        zorder=1
    )
    ax.add_patch(card)

    # Protective dashed halo
    halo = Circle((500, 560), 330, fill=False, edgecolor="#CCFBF1", linewidth=5, linestyle=(0, (10, 10)), zorder=2)
    ax.add_patch(halo)

    # Medical Shield path
    # Shield coordinates (in 0-1000 space, y inverted relative to svg)
    # Top center: (500, 850), top right: (760, 670), tip: (500, 270), top left: (240, 670)
    shield_verts = [
        (500, 840), # start top center
        (650, 840), (760, 780), (760, 650), # top right shoulder
        (760, 450), (620, 340), (500, 260), # right side to bottom tip
        (380, 340), (240, 450), (240, 650), # left side from bottom tip
        (240, 780), (350, 840), (500, 840), # top left shoulder back to top center
    ]
    shield_codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
    ]
    shield_path = Path(shield_verts, shield_codes)
    shield_patch = PathPatch(shield_path, facecolor="#0F766E", edgecolor="#14B8A6", linewidth=8, zorder=3)
    ax.add_patch(shield_patch)

    # Maternal Embracing Heart Outline
    # Centered at (500, 580)
    t = np.linspace(0, 2 * np.pi, 300)
    # Parametric heart
    hx = 16 * (np.sin(t) ** 3)
    hy = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    # Scale and center inside shield
    hx = 500 + hx * 12.0
    hy = 590 + hy * 12.0
    ax.plot(hx, hy, color="#F0FDFA", linewidth=14, solid_capstyle="round", zorder=4)

    # Fetal / Maternal ECG pulse waveform
    ecg_x = [300, 390, 420, 450, 490, 520, 550, 575, 600, 700]
    ecg_y = [570, 570, 520, 660, 440, 620, 540, 570, 570, 570]
    ax.plot(ecg_x, ecg_y, color="#E11D48", linewidth=11, solid_capstyle="round", solid_joinstyle="round", zorder=5)

    # Vitality center spark
    center_glow = Circle((500, 560), 16, facecolor="#FFFFFF", edgecolor="#E11D48", linewidth=4, zorder=6)
    ax.add_patch(center_glow)
    center_core = Circle((500, 560), 8, facecolor="#E11D48", zorder=7)
    ax.add_patch(center_core)

    # Triage cross emblem
    cross_bg = Circle((500, 370), 36, facecolor="#0D9488", edgecolor="#F0FDFA", linewidth=4, zorder=4)
    ax.add_patch(cross_bg)
    ax.plot([478, 522], [370, 370], color="#FFFFFF", linewidth=7, solid_capstyle="round", zorder=5)
    ax.plot([500, 500], [348, 392], color="#FFFFFF", linewidth=7, solid_capstyle="round", zorder=5)

    # Brand Title Typography
    ax.text(
        420, 165, "MATERNA",
        fontsize=42, fontweight="heavy", color="#0F766E",
        ha="right", va="center", family="sans-serif", zorder=4
    )
    ax.text(
        425, 165, "TRIAGE",
        fontsize=42, fontweight="heavy", color="#E11D48",
        ha="left", va="center", family="sans-serif", zorder=4
    )

    ax.text(
        500, 115, "CLINICAL DECISION SUPPORT",
        fontsize=16, fontweight="bold", color="#64748B",
        ha="center", va="center", family="sans-serif", zorder=4
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    # Verify image dimensions with Pillow
    img = Image.open(output_path)
    w, h = img.size
    print(f"[Logo] Successfully generated {output_path} ({w}x{h} px)")
    if (w, h) != (1024, 1024):
        img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        img.save(output_path)
        print(f"[Logo] Resized cleanly to exactly (1024, 1024)")


if __name__ == "__main__":
    render_logo()

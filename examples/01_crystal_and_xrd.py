from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paths import add_src_to_path

add_src_to_path()

from solidstate import face_centered_cubic, powder_peaks_cubic


OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    lattice = face_centered_cubic(a=1.0)
    points = lattice.points(-1, 1)
    reciprocal_points = lattice.reciprocal_points(-2, 2)
    peaks = powder_peaks_cubic(a=4.05, wavelength=1.5406, lattice="fcc", max_index=6)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=35)
    ax1.set_title("FCC real-space points")
    ax1.set_xlabel("x/a")
    ax1.set_ylabel("y/a")
    ax1.set_zlabel("z/a")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(reciprocal_points[:, 0], reciprocal_points[:, 1], reciprocal_points[:, 2], s=15)
    ax2.set_title("Reciprocal lattice points")
    ax2.set_xlabel("kx")
    ax2.set_ylabel("ky")
    ax2.set_zlabel("kz")

    ax3 = fig.add_subplot(1, 3, 3)
    two_theta = [float(peak["two_theta"]) for peak in peaks[:12]]
    intensity = [float(peak["relative_intensity"]) for peak in peaks[:12]]
    labels = ["".join(map(str, peak["hkl"])) for peak in peaks[:12]]
    ax3.vlines(two_theta, 0, intensity, color="tab:red", linewidth=2)
    for x_value, y_value, label in zip(two_theta, intensity, labels):
        ax3.text(x_value, y_value + 0.4, label, ha="center", va="bottom", fontsize=8)
    ax3.set_title("Schematic FCC powder peaks")
    ax3.set_xlabel("2 theta (degrees)")
    ax3.set_ylabel("relative intensity")
    ax3.set_ylim(0, max(intensity) * 1.25)

    fig.tight_layout()
    output = OUTDIR / "01_crystal_and_xrd.png"
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

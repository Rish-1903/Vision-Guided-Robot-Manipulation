"""Create an accurately scaled digital checkerboard image for CoppeliaSim."""
from pathlib import Path
import argparse
import cv2
import numpy as np

INNER_CORNERS = (9, 6)  # number of intersection points per column/row
SQUARE_METERS = 0.03
BORDER_SQUARES = 1
PIXELS_PER_SQUARE = 96


def generate(path: Path) -> tuple[int, int]:
    ncols = INNER_CORNERS[0] + 1
    nrows = INNER_CORNERS[1] + 1
    w_sq = ncols + 2 * BORDER_SQUARES
    h_sq = nrows + 2 * BORDER_SQUARES
    img = np.full((h_sq * PIXELS_PER_SQUARE, w_sq * PIXELS_PER_SQUARE, 3), 255, np.uint8)
    for row in range(nrows):
        for col in range(ncols):
            if (row + col) % 2 == 0:
                y = (row + BORDER_SQUARES) * PIXELS_PER_SQUARE
                x = (col + BORDER_SQUARES) * PIXELS_PER_SQUARE
                img[y:y + PIXELS_PER_SQUARE, x:x + PIXELS_PER_SQUARE] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Could not write {path}")
    return img.shape[1], img.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'checkerboard.png')
    args = parser.parse_args()
    size = generate(args.output)
    print(f"Wrote {args.output.resolve()}: {size[0]} × {size[1]} pixels")
    board_width = (INNER_CORNERS[0] + 1 + 2 * BORDER_SQUARES) * SQUARE_METERS
    board_height = (INNER_CORNERS[1] + 1 + 2 * BORDER_SQUARES) * SQUARE_METERS
    print(f"Use physical CoppeliaSim plane size: {board_width:.3f} × {board_height:.3f} m")

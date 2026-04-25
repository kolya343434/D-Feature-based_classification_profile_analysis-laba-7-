from __future__ import annotations

from collections import deque
from typing import TypedDict

import numpy as np


class Component(TypedDict):
    x0: int
    y0: int
    x1: int
    y1: int
    area: int


def connected_components(binary: np.ndarray) -> tuple[np.ndarray, list[Component]]:
    """
    4-connected components for boolean foreground mask.
    Returns labels array (0 = background) and list of bounding boxes + area.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comps: list[Component] = []
    cur = 0

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            cur += 1
            q: deque[tuple[int, int]] = deque()
            q.append((y, x))
            labels[y, x] = cur

            x0 = x1 = x
            y0 = y1 = y
            area = 0

            while q:
                cy, cx = q.popleft()
                area += 1
                if cx < x0:
                    x0 = cx
                if cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                if cy > y1:
                    y1 = cy

                if cy > 0 and binary[cy - 1, cx] and labels[cy - 1, cx] == 0:
                    labels[cy - 1, cx] = cur
                    q.append((cy - 1, cx))
                if cy + 1 < h and binary[cy + 1, cx] and labels[cy + 1, cx] == 0:
                    labels[cy + 1, cx] = cur
                    q.append((cy + 1, cx))
                if cx > 0 and binary[cy, cx - 1] and labels[cy, cx - 1] == 0:
                    labels[cy, cx - 1] = cur
                    q.append((cy, cx - 1))
                if cx + 1 < w and binary[cy, cx + 1] and labels[cy, cx + 1] == 0:
                    labels[cy, cx + 1] = cur
                    q.append((cy, cx + 1))

            comps.append(
                Component(
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1) + 1,  # exclusive
                    y1=int(y1) + 1,  # exclusive
                    area=int(area),
                )
            )

    return labels, comps


def sort_components_left_to_right(comps: list[Component]) -> list[Component]:
    return sorted(comps, key=lambda c: (c["x0"], c["y0"]))


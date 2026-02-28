#!/usr/bin/env python3
"""
FiltraViz (H0/H1 focused) — Rips ⇄ Clique/Flag complex
- Keep triangles (2-simplices) so H1 "death" (loop filled) is visible.
- UI focuses on H0/H1 (no H2 plots) to avoid empty/noisy UI for 2D toy data.
- Geometry view shows triangle fills with diff coloring:
    BLUE   : triangles already present before the move
    ORANGE : triangles added when ε increases
    PURPLE : triangles removed when ε decreases
- Barcodes: H0, H1 only (+ ε vertical indicator lines)
- Betti curves: β0, β1 only (+ ε vertical indicator line)
- Explain: lists events (H0 deaths, H1 births, H1 deaths) in the ε interval
- Clique mode: k-NN parameter k is controllable in UI

Fix in this version:
- Remove pyqtgraph warning "Item already added to PlotItem, ignoring."
  by ensuring vlines are added ONLY in rebuild() and only if vline.scene() is None.
  on_eps_changed() only updates setValue().

Dependencies:
  pip install pyqt6 pyqtgraph gudhi numpy
"""

import sys
import math
import numpy as np

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QPolygonF
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QGraphicsPolygonItem

import pyqtgraph as pg
import gudhi as gd


# -----------------------------
# Data generators (toy datasets)
# -----------------------------
def make_annulus(n=100, r=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rad = r + rng.normal(0.0, noise, size=n)
    x = rad * np.cos(ang)
    y = rad * np.sin(ang)
    return np.c_[x, y]


def make_two_clusters(n=100, sep=2.5, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    c1 = np.array([-sep / 2.0, 0.0])
    c2 = np.array([+sep / 2.0, 0.0])
    x1 = c1 + rng.normal(0.0, noise, size=(n1, 2))
    x2 = c2 + rng.normal(0.0, noise, size=(n2, 2))
    return np.vstack([x1, x2])


def make_grid_with_noise(m=10, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1.0, 1.0, m)
    ys = np.linspace(-1.0, 1.0, m)
    pts = np.array([[x, y] for x in xs for y in ys], dtype=float)
    pts += rng.normal(0.0, noise, size=pts.shape)
    return pts


# -----------------------------
# Core TDA / persistence helpers
# -----------------------------
def compute_rips_persistence(points: np.ndarray, max_edge_length: float, max_dim: int = 2):
    """
    We compute up to max_dim=2 to obtain triangles (for H1 death),
    but UI will focus on H0/H1.
    """
    rc = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    st = rc.create_simplex_tree(max_dimension=max_dim)
    st.persistence()

    intervals = {}
    for d in range(max_dim + 1):
        inter = st.persistence_intervals_in_dimension(d)
        intervals[d] = np.array(inter, dtype=float) if len(inter) else np.zeros((0, 2), dtype=float)

    edges = []
    triangles = []
    for simplex, filt in st.get_filtration():
        if len(simplex) == 2:
            i, j = simplex
            edges.append((i, j, float(filt)))
        elif len(simplex) == 3:
            i, j, k = simplex
            triangles.append((i, j, k, float(filt)))

    return intervals, edges, triangles


def compute_knn_cost_edges(points: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """Undirected k-NN graph with Euclidean distance as *cost*."""
    n = int(points.shape[0])
    if n <= 1:
        return []
    k = int(max(1, min(k, n - 1)))

    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)

    edges_set: set[tuple[int, int]] = set()
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        nn = np.argpartition(dist[i], kth=k - 1)[:k]
        for j in nn:
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if (a, b) in edges_set:
                continue
            edges_set.add((a, b))
            edges.append((a, b, float(dist[a, b])))
    return edges


def compute_clique_persistence_from_cost_graph(
    n_vertices: int,
    cost_edges: list[tuple[int, int, float]],
    max_dim: int = 2,
):
    """
    Clique(=Flag) complex from a weighted graph.
    Edge filtration = cost.
    Triangle filtration = max(costs of its edges).
    """
    n = int(n_vertices)
    st = gd.SimplexTree()

    for v in range(n):
        st.insert([v], filtration=0.0)

    edge_cost: dict[tuple[int, int], float] = {}
    adj: list[set[int]] = [set() for _ in range(n)]

    edges: list[tuple[int, int, float]] = []
    for i, j, c in cost_edges:
        a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        cc = float(c)
        edge_cost[(a, b)] = cc
        adj[a].add(b)
        adj[b].add(a)
        st.insert([a, b], filtration=cc)
        edges.append((a, b, cc))

    triangles: list[tuple[int, int, int, float]] = []
    if max_dim >= 2:
        for a in range(n):
            na = adj[a]
            for b in na:
                if a >= b:
                    continue
                common = na.intersection(adj[b])
                for c in common:
                    if b >= c:
                        continue
                    cab = edge_cost[(a, b)]
                    cac = edge_cost[(a, c)] if a < c else edge_cost[(c, a)]
                    cbc = edge_cost[(b, c)]
                    filt = float(max(cab, cac, cbc))
                    st.insert([a, b, c], filtration=filt)
                    triangles.append((a, b, c, filt))

    st.make_filtration_non_decreasing()
    st.persistence()

    intervals = {}
    for d in range(max_dim + 1):
        inter = st.persistence_intervals_in_dimension(d)
        intervals[d] = np.array(inter, dtype=float) if len(inter) else np.zeros((0, 2), dtype=float)

    return intervals, edges, triangles


def betti_from_intervals(intervals: np.ndarray, eps: float) -> int:
    if intervals.size == 0:
        return 0
    b = intervals[:, 0]
    d = intervals[:, 1]
    alive = (b <= eps) & ((eps < d) | np.isinf(d))
    return int(np.sum(alive))


def alive_mask(intervals: np.ndarray, eps: float) -> np.ndarray:
    if intervals.size == 0:
        return np.zeros((0,), dtype=bool)
    b = intervals[:, 0]
    d = intervals[:, 1]
    return (b <= eps) & ((eps < d) | np.isinf(d))


def compute_betti_curves(intervals0: np.ndarray, intervals1: np.ndarray, eps_grid: np.ndarray):
    b0 = np.array([betti_from_intervals(intervals0, e) for e in eps_grid], dtype=float)
    b1 = np.array([betti_from_intervals(intervals1, e) for e in eps_grid], dtype=float)
    return b0, b1


# -----------------------------
# Plotting helpers (pyqtgraph)
# -----------------------------
def _barcode_xmax(intervals: np.ndarray, eps_max: float) -> float:
    if intervals.size == 0:
        return max(1.0, float(eps_max) * 1.10 + 1e-9)
    finite_death = intervals[:, 1][~np.isinf(intervals[:, 1])]
    mx = float(np.max(finite_death)) if finite_death.size else 0.0
    base = max(mx, float(eps_max))
    return base * 1.10 + 1e-9


def plot_barcode_items(plot: pg.PlotWidget, intervals: np.ndarray, title: str, eps_max: float):
    plot.clear()
    plot.setTitle(title)
    plot.showGrid(x=True, y=True, alpha=0.2)

    items = []
    xmax = _barcode_xmax(intervals, eps_max)

    if intervals.size == 0:
        plot.setYRange(0, 1)
        plot.setXRange(0, xmax)
        return items

    for idx, (b, d) in enumerate(intervals):
        y = idx
        x0 = float(b)
        x1 = float(d) if not np.isinf(d) else xmax
        item = pg.PlotDataItem([x0, x1], [y, y], pen=pg.mkPen(width=2))
        plot.addItem(item)
        items.append({"idx": idx, "item": item, "birth": float(b), "death": float(d)})

    plot.setYRange(-1, len(intervals) + 1)
    plot.setXRange(0, xmax)
    return items


def plot_persistence_diagram(plot: pg.PlotWidget, intervals0: np.ndarray, intervals1: np.ndarray, title: str):
    plot.clear()
    plot.setTitle(title)
    plot.showGrid(x=True, y=True, alpha=0.2)

    pts0 = intervals0.copy()
    pts1 = intervals1.copy()
    pts0 = pts0[~np.isinf(pts0[:, 1])] if pts0.size else pts0
    pts1 = pts1[~np.isinf(pts1[:, 1])] if pts1.size else pts1

    if pts0.size or pts1.size:
        all_pts = np.vstack([p for p in (pts0, pts1) if p.size])
        mx = float(np.max(all_pts))
    else:
        mx = 1.0

    plot.addItem(
        pg.PlotDataItem([0, mx], [0, mx], pen=pg.mkPen("w", width=1, style=QtCore.Qt.PenStyle.DashLine))
    )

    if pts0.size:
        plot.addItem(pg.ScatterPlotItem(x=pts0[:, 0], y=pts0[:, 1], symbol="o", size=7))
    if pts1.size:
        plot.addItem(pg.ScatterPlotItem(x=pts1[:, 0], y=pts1[:, 1], symbol="x", size=9))

    plot.setXRange(0, mx * 1.05 + 1e-9)
    plot.setYRange(0, mx * 1.05 + 1e-9)
    plot.setLabel("bottom", "birth")
    plot.setLabel("left", "death")


# -----------------------------
# Event / highlight helpers
# -----------------------------
class BarcodeHighlighter:
    """alive: GREEN, dead: GRAY, flash(): YELLOW thick"""
    def __init__(self):
        self.items = []
        self.last_flash_idx = None

    def set_items(self, items):
        self.items = items
        self.last_flash_idx = None

    def set_alive_dead_colors(self, alive: np.ndarray):
        for d in self.items:
            idx = d["idx"]
            is_alive = (idx < len(alive)) and bool(alive[idx])
            pen = pg.mkPen("g" if is_alive else (160, 160, 160), width=2 if is_alive else 1)
            d["item"].setPen(pen)
        if self.last_flash_idx is not None:
            self.flash(self.last_flash_idx)

    def flash(self, idx: int):
        if idx < 0 or idx >= len(self.items):
            return
        self.last_flash_idx = idx
        self.items[idx]["item"].setPen(pg.mkPen("y", width=4))


def collect_crossings(items, a: float, b: float, kind: str):
    """
    Collect all crossings between a and b (either direction).
    kind: "birth" uses birth threshold, "death" uses finite death threshold.
    Returns sorted list of (idx, t).
    """
    if a == b:
        return []
    lo, hi = (a, b) if a < b else (b, a)
    out = []
    for d in items:
        t = d["birth"] if kind == "birth" else d["death"]
        if kind == "death" and np.isinf(t):
            continue
        if lo < t <= hi:
            out.append((d["idx"], float(t)))
    out.sort(key=lambda x: x[1])
    return out


def find_nearest_edge_at(edges, t: float):
    if not edges:
        return None
    best = None
    best_dt = 1e18
    for i, j, f in edges:
        dt = abs(f - t)
        if dt < best_dt:
            best_dt = dt
            best = (i, j, f)
    return best


# -----------------------------
# Geometry drawing (ViewBox coords)
# -----------------------------
class GeometryLayer:
    """
    Base geometry drawn in ViewBox (data coordinates).
    Triangle diff coloring:
      - BLUE   : f <= min(prev_eps, eps)
      - ORANGE : prev_eps < f <= eps (when eps increases)
      - PURPLE : eps < f <= prev_eps (when eps decreases)
    """
    def __init__(self, plot: pg.PlotWidget):
        self.vb = plot.getViewBox()
        self.edge_items = []
        self.tri_old_items = []
        self.tri_new_items = []
        self.tri_removed_items = []

    def clear(self):
        for arr in (self.edge_items, self.tri_old_items, self.tri_new_items, self.tri_removed_items):
            for it in arr:
                try:
                    self.vb.removeItem(it)
                except Exception:
                    pass
            arr.clear()

    def _add_triangle(self, p0, p1, p2, fill_rgba, target_list):
        poly = QPolygonF([QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1])])
        gi = QGraphicsPolygonItem(poly)
        gi.setBrush(pg.mkBrush(*fill_rgba))
        gi.setPen(pg.mkPen(None))
        self.vb.addItem(gi)
        target_list.append(gi)

    def draw(self, points, edges, triangles, eps: float, prev_eps: float):
        self.clear()

        edge_pen = pg.mkPen(180, 180, 180, width=1)
        for i, j, f in edges:
            if f <= eps:
                p0 = points[i]
                p1 = points[j]
                item = pg.PlotDataItem([p0[0], p1[0]], [p0[1], p1[1]], pen=edge_pen)
                self.vb.addItem(item)
                self.edge_items.append(item)

        eps_lo = min(eps, prev_eps)
        for i, j, k, f in triangles:
            if f <= eps_lo:
                self._add_triangle(points[i], points[j], points[k], (80, 160, 255, 60), self.tri_old_items)
            elif eps < prev_eps and (eps < f <= prev_eps):
                self._add_triangle(points[i], points[j], points[k], (180, 80, 255, 70), self.tri_removed_items)
            elif eps >= prev_eps and (prev_eps < f <= eps):
                self._add_triangle(points[i], points[j], points[k], (255, 165, 0, 90), self.tri_new_items)


class GeometryOverlay:
    """Overlay highlight in ViewBox (edge or triangle outlines)"""
    def __init__(self, plot: pg.PlotWidget):
        self.vb = plot.getViewBox()
        self.edge_item = None
        self.tri_outline_items = []

    def clear(self):
        if self.edge_item is not None:
            try:
                self.vb.removeItem(self.edge_item)
            except Exception:
                pass
            self.edge_item = None
        for it in self.tri_outline_items:
            try:
                self.vb.removeItem(it)
            except Exception:
                pass
        self.tri_outline_items = []

    def highlight_edge(self, p0, p1, color="r", width=6):
        self.clear()
        self.edge_item = pg.PlotDataItem([p0[0], p1[0]], [p0[1], p1[1]], pen=pg.mkPen(color, width=width))
        self.vb.addItem(self.edge_item)

    def highlight_triangles_outline(self, tris_pts, color="orange", width=5):
        self.clear()
        for (p0, p1, p2) in tris_pts:
            item = pg.PlotDataItem(
                [p0[0], p1[0], p2[0], p0[0]],
                [p0[1], p1[1], p2[1], p0[1]],
                pen=pg.mkPen(color, width=width),
            )
            self.vb.addItem(item)
            self.tri_outline_items.append(item)


# -----------------------------
# Main GUI
# -----------------------------
class FiltraViz(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FiltraViz: H0/H1 (Rips ⇄ Clique) + Triangle Fill Events")

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        upper = QtWidgets.QWidget()
        upper_layout = QtWidgets.QHBoxLayout(upper)
        root.addWidget(upper, stretch=1)

        # Left: geometry
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        upper_layout.addWidget(left, stretch=2)

        self.geom = pg.PlotWidget()
        self.geom.setAspectLocked(True)
        self.geom.showGrid(x=True, y=True, alpha=0.2)
        self.geom.setTitle("Geometry: OLD(blue), NEW(orange), REMOVED(purple)")
        left_layout.addWidget(self.geom, stretch=1)

        self.scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(230, 230, 230))
        self.geom.getViewBox().addItem(self.scatter)

        self.base_layer = GeometryLayer(self.geom)
        self.overlay = GeometryOverlay(self.geom)

        # Right: tabs
        right = QtWidgets.QTabWidget()
        upper_layout.addWidget(right, stretch=3)

        # Barcodes tab (H0/H1 only)
        tab_bar = QtWidgets.QWidget()
        tab_bar_layout = QtWidgets.QVBoxLayout(tab_bar)
        right.addTab(tab_bar, "Barcodes")
        self.bar0 = pg.PlotWidget()
        self.bar1 = pg.PlotWidget()
        tab_bar_layout.addWidget(self.bar0, stretch=1)
        tab_bar_layout.addWidget(self.bar1, stretch=1)

        # Diagram tab
        tab_diag = QtWidgets.QWidget()
        tab_diag_layout = QtWidgets.QVBoxLayout(tab_diag)
        right.addTab(tab_diag, "Diagram")
        self.pdiag = pg.PlotWidget()
        tab_diag_layout.addWidget(self.pdiag, stretch=1)

        # Betti tab (β0/β1 only)
        tab_betti = QtWidgets.QWidget()
        tab_betti_layout = QtWidgets.QVBoxLayout(tab_betti)
        right.addTab(tab_betti, "Betti")
        self.betti_plot = pg.PlotWidget()
        self.betti_plot.showGrid(x=True, y=True, alpha=0.2)
        tab_betti_layout.addWidget(self.betti_plot, stretch=1)

        # Explain tab
        tab_explain = QtWidgets.QWidget()
        tab_explain_layout = QtWidgets.QVBoxLayout(tab_explain)
        right.addTab(tab_explain, "Explain")
        self.explain_chk = QtWidgets.QCheckBox("Explanation mode (list all events in the ε-interval)")
        self.explain_chk.setChecked(True)
        tab_explain_layout.addWidget(self.explain_chk)
        self.explain_text = QtWidgets.QTextEdit()
        self.explain_text.setReadOnly(True)
        tab_explain_layout.addWidget(self.explain_text, stretch=1)

        # Controls
        controls = QtWidgets.QWidget()
        c = QtWidgets.QHBoxLayout(controls)
        root.addWidget(controls, stretch=0)

        self.dataset = QtWidgets.QComboBox()
        self.dataset.addItems(["Annulus (1 hole)", "Two clusters (H0 merges)", "Grid+noise (many small holes)"])
        c.addWidget(QtWidgets.QLabel("Dataset:"))
        c.addWidget(self.dataset)

        self.complex_mode = QtWidgets.QComboBox()
        self.complex_mode.addItems(["Rips (distance threshold)", "Clique (k-NN cost graph)"])
        self.complex_mode.setCurrentIndex(1)  # default = Clique
        c.addWidget(QtWidgets.QLabel("Complex:"))
        c.addWidget(self.complex_mode)

        self.n_nodes = QtWidgets.QSpinBox()
        self.n_nodes.setRange(10, 2000)
        self.n_nodes.setSingleStep(10)
        self.n_nodes.setValue(100)
        c.addWidget(QtWidgets.QLabel("N:"))
        c.addWidget(self.n_nodes)

        self.k_knn = QtWidgets.QSpinBox()
        self.k_knn.setRange(1, 200)
        self.k_knn.setValue(10)
        c.addWidget(QtWidgets.QLabel("k (Clique):"))
        c.addWidget(self.k_knn)

        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 9999)
        self.seed.setValue(0)
        c.addWidget(QtWidgets.QLabel("Seed:"))
        c.addWidget(self.seed)

        self.rebuild_btn = QtWidgets.QPushButton("Rebuild")
        c.addWidget(self.rebuild_btn)

        self.eps_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.eps_slider.setMinimum(0)
        self.eps_slider.setMaximum(1000)
        self.eps_slider.setValue(120)
        c.addWidget(QtWidgets.QLabel("ε:"))
        c.addWidget(self.eps_slider, stretch=1)

        self.eps_label = QtWidgets.QLabel("ε = 0.000")
        c.addWidget(self.eps_label)

        self.betti_label = QtWidgets.QLabel("β0=0, β1=0")
        c.addWidget(self.betti_label)

        # ε indicator lines (add only in rebuild(); always visible by Z)
        self.vline0 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=2))
        self.vline1 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=2))
        self.vline_betti = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=2))
        for ln in (self.vline0, self.vline1, self.vline_betti):
            ln.setZValue(10_000)

        # persistence data (H0/H1 + geometry)
        self.points = np.zeros((0, 2), dtype=float)
        self.edges = []
        self.triangles = []
        self.intervals0 = np.zeros((0, 2), dtype=float)
        self.intervals1 = np.zeros((0, 2), dtype=float)

        self.bar0_items = []
        self.bar1_items = []
        self.hi0 = BarcodeHighlighter()
        self.hi1 = BarcodeHighlighter()

        self.prev_eps = 0.0
        self.prev_b0 = 0
        self.prev_b1 = 0

        # events
        self.rebuild_btn.clicked.connect(self.rebuild)
        self.dataset.currentIndexChanged.connect(lambda _=None: self.rebuild())
        self.seed.valueChanged.connect(lambda _=None: self.rebuild())
        self.complex_mode.currentIndexChanged.connect(lambda _=None: self.rebuild())
        self.n_nodes.valueChanged.connect(lambda _=None: self.rebuild())
        self.k_knn.valueChanged.connect(lambda _=None: self.rebuild())
        self.eps_slider.valueChanged.connect(self.on_eps_changed)

        self.rebuild()

    def slider_to_eps(self, v: int) -> float:
        return float(v) / 1000.0 * float(self.eps_max)

    def _add_vline_once(self, plot: pg.PlotWidget, vline: pg.InfiniteLine):
        """
        Add vline only if it is not in any scene yet.
        This avoids duplicate-add warnings reliably.
        """
        if vline.scene() is not None:
            return
        plot.addItem(vline)

    def rebuild(self):
        idx = self.dataset.currentIndex()
        seed = int(self.seed.value())
        n = int(self.n_nodes.value())
        mode = self.complex_mode.currentIndex()

        self.k_knn.setEnabled(mode == 1)

        # generate points
        if idx == 0:
            pts = make_annulus(n=n, r=1.0, noise=0.05, seed=seed)
            suggested_max = 1.2
        elif idx == 1:
            pts = make_two_clusters(n=n, sep=2.3, noise=0.18, seed=seed)
            suggested_max = 2.0
        else:
            mgrid = max(3, int(round(math.sqrt(n))))
            pts = make_grid_with_noise(m=mgrid, noise=0.04, seed=seed)
            if pts.shape[0] > n:
                rng = np.random.default_rng(seed)
                sel = rng.choice(pts.shape[0], size=n, replace=False)
                pts = pts[sel]
            suggested_max = 0.8

        self.points = pts

        # compute persistence up to dim=2 (triangles), but we only use H0/H1 for UI
        if mode == 0:
            self.eps_max = float(suggested_max)
            intervals, edges, triangles = compute_rips_persistence(
                points=self.points,
                max_edge_length=self.eps_max,
                max_dim=2,
            )
        else:
            k = int(self.k_knn.value())
            k = min(k, max(1, n - 1))
            cost_edges = compute_knn_cost_edges(self.points, k=k)
            self.eps_max = float(max([c for _, _, c in cost_edges], default=suggested_max))
            intervals, edges, triangles = compute_clique_persistence_from_cost_graph(
                n_vertices=self.points.shape[0],
                cost_edges=cost_edges,
                max_dim=2,
            )

        self.edges = edges
        self.triangles = triangles
        self.intervals0 = intervals.get(0, np.zeros((0, 2), dtype=float))
        self.intervals1 = intervals.get(1, np.zeros((0, 2), dtype=float))

        # scatter
        self.scatter.setData(self.points[:, 0], self.points[:, 1])

        # barcodes (H0/H1)
        self.bar0_items = plot_barcode_items(self.bar0, self.intervals0, "H0 barcode", eps_max=self.eps_max)
        self.bar1_items = plot_barcode_items(self.bar1, self.intervals1, "H1 barcode", eps_max=self.eps_max)

        # add vlines exactly once per scene; safe after plot.clear()
        self._add_vline_once(self.bar0, self.vline0)
        self._add_vline_once(self.bar1, self.vline1)

        self.hi0.set_items(self.bar0_items)
        self.hi1.set_items(self.bar1_items)

        # diagram (H0/H1)
        plot_persistence_diagram(self.pdiag, self.intervals0, self.intervals1, "Persistence diagram (H0: o, H1: x)")

        # betti curves (β0/β1)
        self.betti_plot.clear()
        self.betti_plot.showGrid(x=True, y=True, alpha=0.2)
        self.betti_plot.setTitle("Betti curves β0, β1 vs ε")
        eps_grid = np.linspace(0.0, float(self.eps_max), 250)
        b0, b1 = compute_betti_curves(self.intervals0, self.intervals1, eps_grid)
        self.betti_plot.addItem(pg.PlotDataItem(eps_grid, b0, pen=pg.mkPen("c", width=2)))
        self.betti_plot.addItem(pg.PlotDataItem(eps_grid, b1, pen=pg.mkPen("m", width=2)))

        self._add_vline_once(self.betti_plot, self.vline_betti)

        # reset prev
        self.prev_eps = 0.0
        self.prev_b0 = betti_from_intervals(self.intervals0, self.prev_eps)
        self.prev_b1 = betti_from_intervals(self.intervals1, self.prev_eps)

        self.overlay.clear()
        self.base_layer.clear()

        # render
        self.on_eps_changed(self.eps_slider.value())

    def on_eps_changed(self, v: int):
        eps = self.slider_to_eps(v)
        self.eps_label.setText(f"ε = {eps:.3f}")

        # update ε lines (do NOT addItem here; prevents warning)
        self.vline0.setValue(eps)
        self.vline1.setValue(eps)
        self.vline_betti.setValue(eps)

        # betti numbers
        b0 = betti_from_intervals(self.intervals0, eps)
        b1 = betti_from_intervals(self.intervals1, eps)
        self.betti_label.setText(f"β0={b0}, β1={b1}")

        # geometry draw (diff triangles)
        self.base_layer.draw(self.points, self.edges, self.triangles, eps=eps, prev_eps=self.prev_eps)

        # barcode coloring
        self.hi0.set_alive_dead_colors(alive_mask(self.intervals0, eps))
        self.hi1.set_alive_dead_colors(alive_mask(self.intervals1, eps))

        # events in the interval (supports big slider jumps)
        h0_deaths = collect_crossings(self.bar0_items, self.prev_eps, eps, kind="death")
        h1_births = collect_crossings(self.bar1_items, self.prev_eps, eps, kind="birth")
        h1_deaths = collect_crossings(self.bar1_items, self.prev_eps, eps, kind="death")

        # representative highlight (latest forward / earliest backward)
        self.overlay.clear()
        rep = None  # ("type", idx, t)
        if eps > self.prev_eps:
            candidates = []
            candidates += [("H0 death", idx, t) for (idx, t) in h0_deaths]
            candidates += [("H1 birth", idx, t) for (idx, t) in h1_births]
            candidates += [("H1 death", idx, t) for (idx, t) in h1_deaths]
            if candidates:
                rep = max(candidates, key=lambda x: x[2])
        elif eps < self.prev_eps:
            candidates = []
            candidates += [("H0 death", idx, t) for (idx, t) in h0_deaths]
            candidates += [("H1 birth", idx, t) for (idx, t) in h1_births]
            candidates += [("H1 death", idx, t) for (idx, t) in h1_deaths]
            if candidates:
                rep = min(candidates, key=lambda x: x[2])

        if rep is not None:
            ev_type, idx_rep, t_rep = rep
            if ev_type == "H0 death":
                self.hi0.flash(idx_rep)
                e = find_nearest_edge_at(self.edges, t_rep)
                if e is not None:
                    i, j, _ = e
                    self.overlay.highlight_edge(self.points[i], self.points[j], color="r", width=6)

            elif ev_type == "H1 birth":
                self.hi1.flash(idx_rep)
                e = find_nearest_edge_at(self.edges, t_rep)
                if e is not None:
                    i, j, _ = e
                    self.overlay.highlight_edge(self.points[i], self.points[j], color="g", width=6)

            elif ev_type == "H1 death":
                self.hi1.flash(idx_rep)
                # Outline triangles close to death time within the changed set
                if eps >= self.prev_eps:
                    changed_tris = [(i, j, k, f) for (i, j, k, f) in self.triangles if (self.prev_eps < f <= eps)]
                else:
                    changed_tris = [(i, j, k, f) for (i, j, k, f) in self.triangles if (eps < f <= self.prev_eps)]
                if changed_tris:
                    changed_tris.sort(key=lambda x: abs(x[3] - t_rep))
                    top = changed_tris[: min(8, len(changed_tris))]
                    tris_pts = [(self.points[i], self.points[j], self.points[k]) for (i, j, k, _) in top]
                    self.overlay.highlight_triangles_outline(tris_pts, color="orange", width=5)

        if self.explain_chk.isChecked():
            lines = []
            lines.append(f"Mode: {self.complex_mode.currentText()}")
            if self.complex_mode.currentIndex() == 1:
                lines.append(f"Clique k: {int(self.k_knn.value())}")
            lines.append(f"ε moved: {self.prev_eps:.4f} → {eps:.4f}")
            lines.append(f"β0,β1: ({self.prev_b0},{self.prev_b1}) → ({b0},{b1})")
            lines.append("")

            def fmt_list(name, xs):
                if not xs:
                    return [f"{name}: (none)"]
                s = [f"{name}:"]
                for (idx, t) in xs[:30]:
                    s.append(f"  - idx={idx}, t={t:.6f}")
                if len(xs) > 30:
                    s.append(f"  ... ({len(xs)-30} more)")
                return s

            lines += fmt_list("H0 deaths (component merges)", h0_deaths)
            lines += fmt_list("H1 births (loop appears)", h1_births)
            lines += fmt_list("H1 deaths (loop filled by triangles)", h1_deaths)
            lines.append("")
            lines.append("Geometry diff coloring:")
            lines.append("  - BLUE  : triangles with f ≤ min(prev_eps, eps)")
            lines.append("  - ORANGE: triangles added in this move (ε increased)")
            lines.append("  - PURPLE: triangles removed in this move (ε decreased)")
            self.explain_text.setPlainText("\n".join(lines))

        # update prev
        self.prev_eps = eps
        self.prev_b0 = b0
        self.prev_b1 = b1


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, background=(20, 20, 20), foreground=(230, 230, 230))
    w = FiltraViz()
    w.resize(1500, 820)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

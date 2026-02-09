#!/usr/bin/env python3
import sys
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
def make_annulus(n=90, r=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, n)
    rr = r + rng.normal(0, noise, n)
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    return np.c_[x, y]

def make_two_clusters(n=90, sep=2.3, noise=0.18, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    c1 = np.array([-sep/2, 0.0])
    c2 = np.array([+sep/2, 0.0])
    X1 = c1 + rng.normal(0, noise, (n1, 2))
    X2 = c2 + rng.normal(0, noise, (n2, 2))
    return np.vstack([X1, X2])

def make_grid_with_noise(m=10, noise=0.04, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1, 1, m)
    ys = np.linspace(-1, 1, m)
    pts = np.array([(x, y) for x in xs for y in ys], dtype=float)
    pts += rng.normal(0, noise, pts.shape)
    return pts


# -----------------------------
# TDA core: Rips + persistence
# -----------------------------
def compute_rips_persistence(points: np.ndarray, max_edge_length: float, max_dim: int = 2):
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


def betti_from_intervals(intervals: np.ndarray, eps: float) -> int:
    if intervals.size == 0:
        return 0
    b = intervals[:, 0]
    d = intervals[:, 1]
    alive = (b <= eps) & ((d == np.inf) | (eps < d))
    return int(np.sum(alive))


def alive_mask(intervals: np.ndarray, eps: float) -> np.ndarray:
    """Return boolean mask of alive intervals at eps for an array (k,2)."""
    if intervals.size == 0:
        return np.zeros((0,), dtype=bool)
    b = intervals[:, 0]
    d = intervals[:, 1]
    return (b <= eps) & ((d == np.inf) | (eps < d))


# -----------------------------
# Barcode drawing as items (for highlighting + alive coloring)
# -----------------------------
def plot_barcode_items(plot: pg.PlotWidget, intervals: np.ndarray, title: str):
    """
    Draw barcode as individual PlotDataItem's.
    Returns items: list {birth, death, y, item, is_inf}
    """
    plot.clear()
    plot.setTitle(title)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "ε (filtration)")
    plot.setLabel("left", "interval index")
    plot.setMouseEnabled(x=True, y=False)

    items = []
    if intervals.size == 0:
        return items

    d = intervals[:, 1].copy()
    if np.any(np.isfinite(d)):
        d_cap = float(np.max(d[np.isfinite(d)]) + 0.2)
    else:
        d_cap = 1.0

    d_sort = d.copy()
    d_sort[np.isinf(d_sort)] = d_cap
    order = np.lexsort((d_sort, intervals[:, 0]))
    ints = intervals[order]

    # Draw with a placeholder pen; actual style is set dynamically per eps.
    placeholder_pen = pg.mkPen(width=2)
    for y, (b, de) in enumerate(ints):
        is_inf = bool(np.isinf(de))
        x0 = float(b)
        x1 = float(de) if np.isfinite(de) else d_cap
        it = plot.plot([x0, x1], [y, y], pen=placeholder_pen)
        items.append({"birth": float(b), "death": float(de), "y": int(y), "item": it, "is_inf": is_inf})

    plot.setYRange(-1, len(ints) + 1, padding=0.02)
    return items


# -----------------------------
# Persistence diagram
# -----------------------------
def plot_persistence_diagram(plot: pg.PlotWidget, intervals0: np.ndarray, intervals1: np.ndarray, title: str):
    plot.clear()
    plot.setTitle(title)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "birth")
    plot.setLabel("left", "death")
    plot.setMouseEnabled(x=True, y=True)

    finite = []
    for ints in (intervals0, intervals1):
        if ints.size:
            d = ints[:, 1]
            finite.extend(d[np.isfinite(d)].tolist())

    max_finite = max(finite) if finite else 1.0
    diag_max = max_finite * 1.05 + 1e-9

    plot.plot([0.0, diag_max], [0.0, diag_max], pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))

    if intervals0.size:
        b0 = intervals0[:, 0]
        d0 = intervals0[:, 1].copy()
        d0[np.isinf(d0)] = diag_max
        plot.plot(b0, d0, pen=None, symbol='o', symbolSize=7)

    if intervals1.size:
        b1 = intervals1[:, 0]
        d1 = intervals1[:, 1].copy()
        d1[np.isinf(d1)] = diag_max
        plot.plot(b1, d1, pen=None, symbol='x', symbolSize=8)

    plot.setXRange(0.0, diag_max, padding=0.02)
    plot.setYRange(0.0, diag_max, padding=0.02)


# -----------------------------
# Explanation (English)
# -----------------------------
def explain_transition(prev_eps, eps, prev_b0, b0, prev_b1, b1, intervals1):
    if prev_eps is None:
        return "Move ε to grow the complex. Watch how β0 (components) and β1 (holes) change."
    if eps < prev_eps:
        return "ε decreased: the complex shrinks (reverse filtration). Betti numbers may increase again."

    msgs = []
    if b0 < prev_b0:
        msgs.append("β0 decreased: two connected components merged (typically when a new edge connects them).")
    if b1 > prev_b1:
        msgs.append("β1 increased: a new 1D cycle (a 'hole') was born (edges completed a loop before it got filled).")
    if b1 < prev_b1:
        msgs.append("β1 decreased: a cycle died (most commonly because 2-simplices filled the loop).")
    if not msgs:
        msgs.append("No Betti change: the complex is growing, but topology stayed the same in this ε range.")
    return " ".join(msgs)


# -----------------------------
# Highlight controllers
# -----------------------------
class BarcodeHighlighter:
    """
    Flash highlighting a single interval index.
    Important: after flash ends, it calls restore_fn() to re-apply alive/dead coloring.
    """
    def __init__(self, parent, restore_fn, highlight_ms=650, highlight_width=6, highlight_color=(0, 0, 0)):
        self.items = []
        self._idx = None
        self._restore_fn = restore_fn
        self._pen_hi = pg.mkPen(color=highlight_color, width=highlight_width)

        self._timer = QtCore.QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._ms = int(highlight_ms)

    def set_items(self, items):
        self.items = items or []
        self._idx = None

    def _on_timeout(self):
        # restore full style based on alive/dead at current eps
        self._idx = None
        self._restore_fn()

    def flash(self, idx):
        if not (0 <= idx < len(self.items)):
            return
        self._idx = idx
        self.items[idx]["item"].setPen(self._pen_hi)
        self._timer.start(self._ms)


class GeometryHighlighter:
    """
    Flash-highlight a single edge (as PlotDataItem) or a single triangle (as QGraphicsPolygonItem).
    """
    def __init__(self, parent, plot_widget: pg.PlotWidget, highlight_ms=650):
        self.plot = plot_widget
        self._timer = QtCore.QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.clear)
        self._ms = int(highlight_ms)

        self.edge_overlay = pg.PlotDataItem(pen=pg.mkPen(width=6))
        self.plot.addItem(self.edge_overlay)

        self.tri_overlay = None
        self._tri_brush = pg.mkBrush(150, 150, 150, 170)
        self._tri_pen = pg.mkPen(width=3)

    def clear(self):
        self.edge_overlay.setData([], [])
        if self.tri_overlay is not None:
            vb = self.plot.getPlotItem().vb
            vb.removeItem(self.tri_overlay)
            self.tri_overlay = None

    def flash_edge(self, p0, p1):
        self.clear()
        self.edge_overlay.setData([p0[0], p1[0]], [p0[1], p1[1]])
        self._timer.start(self._ms)

    def flash_triangle(self, p0, p1, p2):
        self.clear()
        vb = self.plot.getPlotItem().vb
        poly = QPolygonF([QPointF(float(p0[0]), float(p0[1])),
                          QPointF(float(p1[0]), float(p1[1])),
                          QPointF(float(p2[0]), float(p2[1]))])
        item = QGraphicsPolygonItem(poly)
        item.setBrush(self._tri_brush)
        item.setPen(self._tri_pen)
        vb.addItem(item)
        self.tri_overlay = item
        self._timer.start(self._ms)


def detect_event_crossing(items, prev_eps, eps, kind="death"):
    """
    Detect next crossed event in (prev_eps, eps].
    Returns (idx, threshold_value) or (None, None).
    """
    if prev_eps is None or eps <= prev_eps:
        return None, None

    cand = []
    if kind == "birth":
        for idx, rec in enumerate(items):
            b = rec["birth"]
            if np.isfinite(b) and (prev_eps < b <= eps + 1e-12):
                cand.append((b, idx))
    elif kind == "death":
        for idx, rec in enumerate(items):
            if rec.get("is_inf", False):
                continue
            d = rec["death"]
            if np.isfinite(d) and (prev_eps < d <= eps + 1e-12):
                cand.append((d, idx))
    else:
        raise ValueError("kind must be 'birth' or 'death'")

    if not cand:
        return None, None
    cand.sort(key=lambda x: x[0])
    thr, idx = cand[0]
    return idx, float(thr)


def find_nearest_edge_at(edges, target_eps, window=(0.0, np.inf)):
    lo, hi = window
    best = None
    best_gap = np.inf
    for i, j, f in edges:
        if not (lo < f <= hi + 1e-12):
            continue
        gap = abs(f - target_eps)
        if gap < best_gap:
            best_gap = gap
            best = (i, j, f)
    return best


def find_nearest_triangle_at(triangles, target_eps, window=(0.0, np.inf)):
    lo, hi = window
    best = None
    best_gap = np.inf
    for i, j, k, f in triangles:
        if not (lo < f <= hi + 1e-12):
            continue
        gap = abs(f - target_eps)
        if gap < best_gap:
            best_gap = gap
            best = (i, j, k, f)
    return best


# -----------------------------
# Betti curves
# -----------------------------
def compute_betti_curve(intervals: np.ndarray, eps_grid: np.ndarray) -> np.ndarray:
    if intervals.size == 0:
        return np.zeros_like(eps_grid, dtype=int)
    b = intervals[:, 0]
    d = intervals[:, 1]
    out = np.empty_like(eps_grid, dtype=int)
    for t, eps in enumerate(eps_grid):
        alive = (b <= eps) & ((d == np.inf) | (eps < d))
        out[t] = int(np.sum(alive))
    return out


# -----------------------------
# Main GUI
# -----------------------------
class TDAToy(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TDA Toy Tool: Filtration ↔ (H0,H1) Barcodes/Diagram/Betti (Rips, 2D)")

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
        self.geom.setTitle("Point cloud + Rips complex at ε")
        left_layout.addWidget(self.geom, stretch=1)

        # Right: tabs
        right = QtWidgets.QTabWidget()
        upper_layout.addWidget(right, stretch=2)

        # Tab: barcodes
        barcode_tab = QtWidgets.QWidget()
        barcode_layout = QtWidgets.QVBoxLayout(barcode_tab)
        self.bar0 = pg.PlotWidget()
        self.bar1 = pg.PlotWidget()
        barcode_layout.addWidget(self.bar0, stretch=1)
        barcode_layout.addWidget(self.bar1, stretch=1)
        right.addTab(barcode_tab, "Barcodes")

        # Tab: persistence diagram
        diag_tab = QtWidgets.QWidget()
        diag_layout = QtWidgets.QVBoxLayout(diag_tab)
        self.pdiag = pg.PlotWidget()
        diag_layout.addWidget(self.pdiag, stretch=1)
        right.addTab(diag_tab, "Persistence diagram")

        # Tab: Betti curves (NEW)
        betti_tab = QtWidgets.QWidget()
        betti_layout = QtWidgets.QVBoxLayout(betti_tab)
        self.betti0_plot = pg.PlotWidget()
        self.betti1_plot = pg.PlotWidget()
        betti_layout.addWidget(self.betti0_plot, stretch=1)
        betti_layout.addWidget(self.betti1_plot, stretch=1)
        right.addTab(betti_tab, "Betti curves")

        # Controls row
        controls = QtWidgets.QWidget()
        c = QtWidgets.QHBoxLayout(controls)
        root.addWidget(controls, stretch=0)

        self.dataset = QtWidgets.QComboBox()
        self.dataset.addItems(["Annulus (1 hole)", "Two clusters (H0 merges)", "Grid+noise (many small holes)"])
        c.addWidget(QtWidgets.QLabel("Dataset:"))
        c.addWidget(self.dataset)

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

        self.eps_label = QtWidgets.QLabel("ε = 0.0")
        c.addWidget(self.eps_label)

        self.betti_label = QtWidgets.QLabel("β0=?, β1=?")
        c.addWidget(self.betti_label)

        self.explain_toggle = QtWidgets.QCheckBox("Explanation mode")
        self.explain_toggle.setChecked(True)
        c.addWidget(self.explain_toggle)

        self.explain = QtWidgets.QTextEdit()
        self.explain.setReadOnly(True)
        self.explain.setMinimumHeight(70)
        root.addWidget(self.explain, stretch=0)

        # State
        self.points = None
        self.edges = []
        self.triangles = []
        self.intervals = {0: np.zeros((0, 2)), 1: np.zeros((0, 2))}
        self.eps_max = 1.0

        self.prev_eps = None
        self.prev_b0 = None
        self.prev_b1 = None

        # Geometry items
        self.scatter = pg.ScatterPlotItem(size=7)
        self.geom.addItem(self.scatter)
        self.edge_item = pg.PlotDataItem(pen=pg.mkPen(width=1))
        self.geom.addItem(self.edge_item)

        # Filled triangles
        self.tri_items = []
        self.tri_brush = pg.mkBrush(150, 150, 150, 80)
        self.tri_pen = pg.mkPen(width=1)

        # Geometry overlay highlighter
        self.geom_hi = GeometryHighlighter(self, self.geom, highlight_ms=650)

        # Barcode vlines
        self.vline0 = pg.InfiniteLine(angle=90, movable=False)
        self.vline1 = pg.InfiniteLine(angle=90, movable=False)

        # Betti vlines (NEW)
        self.vline_b0 = pg.InfiniteLine(angle=90, movable=False)
        self.vline_b1 = pg.InfiniteLine(angle=90, movable=False)
        self.betti0_plot.addItem(self.vline_b0)
        self.betti1_plot.addItem(self.vline_b1)

        # Barcode items
        self.bar0_items = []
        self.bar1_items = []

        # Pens for alive/dead bars (NEW)
        self.pen_alive0 = pg.mkPen(color=(40, 90, 200), width=3)  # H0 alive
        self.pen_dead0  = pg.mkPen(color=(180, 180, 180), width=2)
        self.pen_alive1 = pg.mkPen(color=(200, 90, 40), width=3)  # H1 alive
        self.pen_dead1  = pg.mkPen(color=(180, 180, 180), width=2)

        # Highlighters (restore_fn re-applies alive/dead coloring)
        self.hi0 = BarcodeHighlighter(self, restore_fn=self.update_alive_bar_styles, highlight_ms=650,
                                      highlight_width=6, highlight_color=(0, 0, 0))
        self.hi1 = BarcodeHighlighter(self, restore_fn=self.update_alive_bar_styles, highlight_ms=650,
                                      highlight_width=6, highlight_color=(0, 0, 0))

        # Betti curves cached (NEW)
        self.betti_eps_grid = None
        self.betti0_vals = None
        self.betti1_vals = None
        self.betti0_curve_item = None
        self.betti1_curve_item = None

        # Signals
        self.rebuild_btn.clicked.connect(self.rebuild)
        self.dataset.currentIndexChanged.connect(self.rebuild)
        self.seed.valueChanged.connect(self.rebuild)
        self.eps_slider.valueChanged.connect(self.on_eps_changed)

        self.rebuild()

    def slider_to_eps(self, v: int) -> float:
        return (v / 1000.0) * self.eps_max

    # -------------------------
    # Geometry rendering
    # -------------------------
    def clear_triangle_items(self):
        vb = self.geom.getPlotItem().vb
        for it in self.tri_items:
            vb.removeItem(it)
        self.tri_items.clear()

    def update_complex_view(self, eps: float):
        pts = self.points

        xs, ys = [], []
        for i, j, f in self.edges:
            if f <= eps + 1e-12:
                xs.extend([pts[i, 0], pts[j, 0], np.nan])
                ys.extend([pts[i, 1], pts[j, 1], np.nan])
        self.edge_item.setData(np.array(xs), np.array(ys)) if xs else self.edge_item.setData([], [])

        self.clear_triangle_items()
        vb = self.geom.getPlotItem().vb
        for i, j, k, f in self.triangles:
            if f <= eps + 1e-12:
                poly = QPolygonF([
                    QPointF(float(pts[i, 0]), float(pts[i, 1])),
                    QPointF(float(pts[j, 0]), float(pts[j, 1])),
                    QPointF(float(pts[k, 0]), float(pts[k, 1])),
                ])
                item = QGraphicsPolygonItem(poly)
                item.setBrush(self.tri_brush)
                item.setPen(self.tri_pen)
                vb.addItem(item)
                self.tri_items.append(item)

    # -------------------------
    # Alive/dead styling (NEW)
    # -------------------------
    def update_alive_bar_styles(self):
        """
        Apply alive/dead pens to all H0/H1 barcode items at current eps.
        Called on eps change and after flash highlight ends.
        """
        eps = self.slider_to_eps(int(self.eps_slider.value()))

        # H0
        for rec in self.bar0_items:
            b, d = rec["birth"], rec["death"]
            alive = (b <= eps) and ((d == np.inf) or (eps < d))
            rec["item"].setPen(self.pen_alive0 if alive else self.pen_dead0)

        # H1
        for rec in self.bar1_items:
            b, d = rec["birth"], rec["death"]
            alive = (b <= eps) and ((d == np.inf) or (eps < d))
            rec["item"].setPen(self.pen_alive1 if alive else self.pen_dead1)

    # -------------------------
    # Betti curves tab (NEW)
    # -------------------------
    def build_betti_curves(self):
        # sample eps grid
        n = 300
        self.betti_eps_grid = np.linspace(0.0, self.eps_max, n)
        self.betti0_vals = compute_betti_curve(self.intervals[0], self.betti_eps_grid)
        self.betti1_vals = compute_betti_curve(self.intervals[1], self.betti_eps_grid)

        # plot
        for pl, title in [(self.betti0_plot, "β0(ε)"), (self.betti1_plot, "β1(ε)")]:
            pl.clear()
            pl.showGrid(x=True, y=True, alpha=0.2)
            pl.setLabel("bottom", "ε (filtration)")
            pl.setLabel("left", "Betti")
            pl.setTitle(title)
            pl.setMouseEnabled(x=True, y=True)

        self.betti0_curve_item = self.betti0_plot.plot(self.betti_eps_grid, self.betti0_vals, pen=pg.mkPen(width=2))
        self.betti1_curve_item = self.betti1_plot.plot(self.betti_eps_grid, self.betti1_vals, pen=pg.mkPen(width=2))

        # re-add vlines
        self.betti0_plot.addItem(self.vline_b0)
        self.betti1_plot.addItem(self.vline_b1)

    # -------------------------
    # Rebuild
    # -------------------------
    def rebuild(self):
        idx = self.dataset.currentIndex()
        seed = int(self.seed.value())

        if idx == 0:
            pts = make_annulus(n=90, r=1.0, noise=0.05, seed=seed)
            suggested_max = 1.2
        elif idx == 1:
            pts = make_two_clusters(n=90, sep=2.3, noise=0.18, seed=seed)
            suggested_max = 2.0
        else:
            pts = make_grid_with_noise(m=10, noise=0.04, seed=seed)
            suggested_max = 0.8

        self.points = pts
        self.eps_max = float(suggested_max)

        intervals, edges, triangles = compute_rips_persistence(
            points=self.points,
            max_edge_length=self.eps_max,
            max_dim=2
        )
        self.intervals[0] = intervals.get(0, np.zeros((0, 2)))
        self.intervals[1] = intervals.get(1, np.zeros((0, 2)))
        self.edges = edges
        self.triangles = triangles

        self.scatter.setData(self.points[:, 0], self.points[:, 1])

        # Barcodes
        self.bar0_items = plot_barcode_items(self.bar0, self.intervals[0], "H0 barcode")
        self.bar1_items = plot_barcode_items(self.bar1, self.intervals[1], "H1 barcode")
        self.bar0.addItem(self.vline0)
        self.bar1.addItem(self.vline1)

        self.hi0.set_items(self.bar0_items)
        self.hi1.set_items(self.bar1_items)

        # Diagram
        plot_persistence_diagram(self.pdiag, self.intervals[0], self.intervals[1], "Persistence diagram (H0: o, H1: x)")

        # Betti curves
        self.build_betti_curves()

        # Reset prev state
        self.prev_eps = None
        self.prev_b0 = None
        self.prev_b1 = None
        self.geom_hi.clear()

        self.on_eps_changed(self.eps_slider.value())
        self.geom.autoRange()

    # -------------------------
    # Main update on eps change
    # -------------------------
    def on_eps_changed(self, v: int):
        eps = self.slider_to_eps(int(v))
        self.eps_label.setText(f"ε = {eps:.3f}")

        # vlines
        self.vline0.setPos(eps)
        self.vline1.setPos(eps)
        self.vline_b0.setPos(eps)
        self.vline_b1.setPos(eps)

        # Betti numbers
        b0 = betti_from_intervals(self.intervals[0], eps)
        b1 = betti_from_intervals(self.intervals[1], eps)
        self.betti_label.setText(f"β0={b0}, β1={b1}")

        # Update alive/dead bar coloring (NEW)
        self.update_alive_bar_styles()

        # Event-driven flash + geometry highlight (forward only)
        if self.prev_eps is not None and eps > self.prev_eps:
            window = (self.prev_eps, eps)

            # H0: DEATH when components merge
            if self.prev_b0 is not None and b0 < self.prev_b0:
                idx0, t0 = detect_event_crossing(self.bar0_items, self.prev_eps, eps, kind="death")
                if idx0 is not None:
                    self.hi0.flash(idx0)
                    e = find_nearest_edge_at(self.edges, t0, window=window)
                    if e is not None:
                        i, j, _ = e
                        self.geom_hi.flash_edge(self.points[i], self.points[j])

            # H1: BIRTH when a hole is born
            if self.prev_b1 is not None and b1 > self.prev_b1:
                idx1b, t1b = detect_event_crossing(self.bar1_items, self.prev_eps, eps, kind="birth")
                if idx1b is not None:
                    self.hi1.flash(idx1b)
                    e = find_nearest_edge_at(self.edges, t1b, window=window)
                    if e is not None:
                        i, j, _ = e
                        self.geom_hi.flash_edge(self.points[i], self.points[j])

            # H1: DEATH when a hole dies
            if self.prev_b1 is not None and b1 < self.prev_b1:
                idx1d, t1d = detect_event_crossing(self.bar1_items, self.prev_eps, eps, kind="death")
                if idx1d is not None:
                    self.hi1.flash(idx1d)
                    tri = find_nearest_triangle_at(self.triangles, t1d, window=window)
                    if tri is not None:
                        i, j, k, _ = tri
                        self.geom_hi.flash_triangle(self.points[i], self.points[j], self.points[k])

        # Explanation
        if self.explain_toggle.isChecked():
            self.explain.setPlainText(
                explain_transition(self.prev_eps, eps, self.prev_b0, b0, self.prev_b1, b1, self.intervals[1])
            )
        else:
            self.explain.setPlainText("")

        # Geometry at eps
        self.update_complex_view(eps)

        # Store prev
        self.prev_eps = eps
        self.prev_b0 = b0
        self.prev_b1 = b1


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = TDAToy()
    w.resize(1450, 860)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

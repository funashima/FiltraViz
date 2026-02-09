# FiltraViz

**Interactive visualization tool for understanding Persistent Homology**

FiltraViz is an educational visualization tool designed to make **Topological Data Analysis (TDA)** intuitive and interactive.
It links **geometry**, **filtration**, and **persistent homology** in real time.

> Move ε → the complex grows → topology changes → barcodes and Betti numbers react.

This tool is especially useful for students encountering **persistent homology** for the first time.

---

## ✨ Features

### 🔹 Interactive Filtration

A slider controls the filtration parameter **ε**, letting you watch the simplicial complex grow continuously.

### 🔹 Linked Views

| View                    | What you see                                            |
| ----------------------- | ------------------------------------------------------- |
| **Geometry View**       | Point cloud, edges, and filled triangles (Rips complex) |
| **Barcodes (H₀ / H₁)**  | Persistent intervals that light up when alive           |
| **Persistence Diagram** | Birth–death scatter plot                                |
| **Betti Curves**        | β₀(ε) and β₁(ε) as functions of ε                       |

All views are synchronized.

---

### 🔹 Event Highlighting (Key Teaching Feature)

When a topological event happens, FiltraViz highlights **both algebra and geometry**:

| Event                       | Barcode          | Geometry                      |
| --------------------------- | ---------------- | ----------------------------- |
| Components merge (H₀ death) | Interval flashes | Responsible **edge** flashes  |
| Hole is born (H₁ birth)     | Interval flashes | Loop-closing **edge** flashes |
| Hole is filled (H₁ death)   | Interval flashes | Filling **triangle** flashes  |

This connects:

> “Something changed in the barcode” → “This simplex caused it”

---

### 🔹 Alive Interval Coloring

Bars currently alive at the chosen ε are highlighted, helping students understand:

> **Persistence = how long a feature lives**

---

### 🔹 Built-in Toy Datasets

| Dataset          | Demonstrates              |
| ---------------- | ------------------------- |
| **Annulus**      | Clear H₁ hole lifecycle   |
| **Two Clusters** | H₀ component merging      |
| **Noisy Grid**   | Many short-lived features |

---

## 🎨 Color Semantics (How to Read the Visuals)

FiltraViz uses color intentionally to communicate topological state.

### 🔹 Barcodes

| Color                 | Meaning                                                |
| --------------------- | ------------------------------------------------------ |
| **Blue (H₀)**         | Connected component intervals                          |
| **Orange (H₁)**       | Hole (1-cycle) intervals                               |
| **Bright / Thick**    | Interval is **alive** at current ε                     |
| **Light Gray / Thin** | Interval is **dead** (feature has disappeared)         |
| **Black Flash**       | A topological **event just occurred** (birth or death) |

This helps students visually connect:

> “Persistent = long and still alive”

---

### 🔹 Geometry View

| Element                  | Color Meaning                             |
| ------------------------ | ----------------------------------------- |
| **Points**               | Sampled data points (neutral)             |
| **Edges**                | 1-simplices currently in the Rips complex |
| **Filled Triangles**     | 2-simplices currently in the complex      |
| **Thick Flashing Edge**  | Edge responsible for a topological event  |
| **Highlighted Triangle** | Triangle that filled a hole (H₁ death)    |

When a barcode interval flashes, a corresponding simplex in the geometry view flashes too.

---

### 🔹 Betti Curves

| Curve             | Meaning                              |
| ----------------- | ------------------------------------ |
| **β₀(ε)**         | Number of connected components       |
| **β₁(ε)**         | Number of holes                      |
| **Vertical Line** | Current ε position in the filtration |

---

### 🧠 Design Philosophy

Color is used to represent **topological state**, not aesthetics:

* **Hue** → homology dimension (H₀ vs H₁)
* **Brightness/Thickness** → alive vs dead
* **Flash** → discrete topological event

This encoding helps learners build intuition without reading equations.

---

## 🧠 Educational Purpose

FiltraViz is built specifically for:

* Lectures on **Persistent Homology**
* Introductory **Topological Data Analysis**
* Visual explanation of:

  * Filtrations
  * Betti numbers
  * Birth and death of topological features

It emphasizes **intuition over formalism**, making abstract concepts observable.

---

## 🛠 Installation

Requires Python 3.9+.

```bash
pip install PyQt6 pyqtgraph numpy gudhi
```

---

## ▶ Running

```bash
python filtraviz.py
```

---

## 🎮 How to Use

1. Choose a dataset
2. Move the **ε slider**
3. Watch:

   * The complex grow
   * Bars light up and die
   * Betti curves change
   * Geometry flash when events occur

Try the **Annulus** dataset to clearly see the life cycle of a 1-dimensional hole.

---

## 🧩 Dependencies

* **PyQt6** — GUI framework
* **PyQtGraph** — fast scientific plotting
* **GUDHI** — persistent homology computation
* **NumPy** — numerical operations

---

## 📚 Concepts Illustrated

* Vietoris–Rips filtration
* Persistent homology
* Betti numbers β₀ and β₁
* Birth and death of topological features
* Relationship between simplices and homological events

---

## 🎓 Intended Audience

Students and instructors in:

* Topological Data Analysis
* Computational topology
* Applied mathematics
* Data science education

---

## 🚀 Future Ideas

* H₂ visualization (3D datasets)
* Animation playback of filtration
* Export figures for lecture slides
* More datasets (real-world point clouds)

---

## 📄 License

MIT License — free to use for teaching and research.

---

## 🤝 Acknowledgements

Persistent homology computations are powered by **GUDHI**.
This project was developed as a teaching aid to make TDA more approachable and visual.

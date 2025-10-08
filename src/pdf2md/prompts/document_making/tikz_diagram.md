# TikZ for Diagrams

Try writing diagrams with TikZ.

## Available Packages

amsmath. amstext. amsfonts. amssymb. array.
tikz-cd. circuitikz. chemfig. pgfplots. tikz-3dplot.

**IMPORTANT:** Don't use TikZ for matrices. Use Typst's `mat` command instead.

## Complete Example

Below is a complete example of a TikZ diagram.
In your output, you should enclose the diagram in a code block with the language `tikz`, too.

**An x-y plot example:**

```tikz
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}

\begin{document}
\begin{tikzpicture}
\begin{axis}[
    xlabel={$t$},
    ylabel={$N$},
    axis lines=center,
    xmin=-0.5, xmax=2.5,
    ymin=-0.5, ymax=4,
    grid=none,
    width=5cm,
    height=8cm,
    every axis x label/.style={at={(current axis.right of origin)}, anchor=west},
    every axis y label/.style={at={(current axis.above origin)}, anchor=south},
    axis line style={->,thick}
]

% Exponential growth (k > 0)
\addplot[
    domain=0:2,
    samples=100,
    thick,
    black
] {exp(0.8*x)} node[pos=0.7, above right] {$k > 0$};

% Constant (k = 0)
\addplot[
    domain=0:2,
    thick,
    black
] {1} node[pos=0.8, above] {$k = 0$};

% Exponential decay (k < 0)
\addplot[
    domain=0:2,
    samples=100,
    thick,
    black
] {exp(-0.8*x)} node[pos=0.6, below right] {$k < 0$};

\end{axis}

% Figure caption
\node[below] at (current bounding box.south) {\textbf{Figure 1:} Graphs of $N(t)$, for different values of $k$};

\end{tikzpicture}
\end{document}
```

**A graph with arrows example:**

```tikz
\begin{document}
\begin{tikzpicture}[->, >=stealth, auto, node distance=3cm, thick, main node/.style={circle,fill=blue!20,draw,font=\sffamily\Large\bfseries}]

  \node[main node] (1) {1};
  \node[main node] (2) [below left of=1, yshift=-1cm] {2};
  \node[main node] (3) [below right of=1, yshift=-1cm] {3};

  \path[every node/.style={font=\sffamily\small}]
    (1) edge [bend left, red] node[above left] {0.6} (2)
    (2) edge [bend left] node[below left] {0.5} (1)
    (1) edge [bend right] node[above right] {0.4} (3)
    (3) edge [bend right, red] node[below right] {0.7} (1)
    (2) edge [bend left] node[below] {0.5} (3)
    (3) edge [bend left] node[above] {0.3} (2);

  \node at (4,-1) [scale=2] {$d=1$};
\end{tikzpicture}
\end{document}
```

**Plotting some function example**:

```tikz
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    xlabel={$x$},
    axis lines=left,
    xmin=0, xmax=2.5,
    ymin=0, ymax=1.7,
    xtick={0, 0.5, 1, 1.5, 2, 2.5},
    ytick={0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6},
    grid=none,
    width=10cm,
    height=8cm,
    every axis x label/.style={at={(current axis.right of origin)}, anchor=north west},
    axis line style={-},
]
% Function from Question 5
% N(t) = (e^(3t) * sech(3t)) / ( (0.2/3) * ln(e^(6t)+1) + 1 - (0.2/3)*ln(2) )
% sech(x) = 1/cosh(x)
\addplot[
    domain=0:2.5,
    samples=200,
    smooth,
    black,
    thick
] { exp(3*x) / (cosh(3*x) * ( (0.2/3) * ln(exp(6*x)+1) + 1 - (0.2/3)*ln(2) )) };
\end{axis}
\node[below=0.5cm] at (current bounding box.south) {\textbf{Figure 2:} Graph of $N(t)$ (Question 5).};
\end{tikzpicture}
\end{document}
```
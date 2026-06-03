from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "poster_assets"
ASSET_DIR.mkdir(exist_ok=True)


PREAMBLE = r"""
\documentclass[border=8pt]{standalone}
\usepackage{amsmath,amssymb,booktabs,array}
\usepackage{xcolor}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{uwpurple}{HTML}{4B2E83}
\definecolor{textdark}{HTML}{25313D}
\begin{document}
"""


def render(name: str, body: str, dpi: int = 260) -> Path:
    tex = ASSET_DIR / f"{name}.tex"
    tex.write_text(PREAMBLE + body + "\n\\end{document}\n", encoding="utf-8")
    subprocess.run(
        ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode", tex.name],
        cwd=ASSET_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    pdf = ASSET_DIR / f"{name}.pdf"
    cropped = ASSET_DIR / f"{name}-crop.pdf"
    subprocess.run(
        ["/Library/TeX/texbin/pdfcrop", pdf.name, cropped.name],
        cwd=ASSET_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["/opt/homebrew/bin/pdftoppm", "-png", "-r", str(dpi), cropped.name, name],
        cwd=ASSET_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return ASSET_DIR / f"{name}-1.png"


render(
    "methods_equations",
    r"""
\begin{minipage}{6.8in}
\color{textdark}
\large
\textbf{\color{uwpurple} Length-aware shaping.}
For prompt group rewards $r_i$, response lengths $L_i$, correct set
$C=\{i:r_i>0\}$, and $L_{\min}=\min_{j\in C} L_j$, incorrect responses keep raw reward.
\[
\begin{aligned}
\textbf{Linear:}\quad
r'_i &= r_i\max\!\left(0,1-\alpha\frac{L_i-L_{\min}}{L_{\min}}\right) \\[2mm]
\textbf{Exponential:}\quad
r'_i &= r_i\exp\!\left(-\lambda\frac{L_i-L_{\min}}{L_{\min}}\right) \\[2mm]
\textbf{Ranked:}\quad
r'_i &= \frac{r_i}{1+\operatorname{Rank}(L_i)} \\[2mm]
\textbf{Binary shortest:}\quad
r'_i &= r_i\,\mathbf{1}[L_i=L_{\min}] \\[2mm]
\textbf{Soft threshold:}\quad
T &= \operatorname{median}\{L_j:j\in C\},\qquad
r'_i =
\begin{cases}
r_i, & L_i\le T\\
r_i\max\!\left(0,1-\frac{L_i-T}{T}\right), & L_i>T
\end{cases}
\end{aligned}
\]
\vspace{1mm}

\textbf{\color{uwpurple} Warm-up.}
\[
r_i^{\mathrm{train}}=(1-w_t)r_i+w_t r'_i,\qquad
w_t=\min\!\left(1,\frac{\text{num\_steps}}{\text{num\_total\_steps}}\right).
\]
Alternatively, solve-rate warm-up keeps $w_t=0$ until the batch solve rate first
hits a chosen threshold; once it has hit, $w_t=1$ for the rest of training.
\end{minipage}
""",
)

render(
    "gfpo_equations",
    r"""
\begin{minipage}{6.7in}
\color{textdark}
\large
\textbf{\color{uwpurple} GFPO baseline (Shrivastava et al., 2025).}
GFPO oversamples $G$ responses, keeps a top-$k$ subset $S$, and zeroes
gradient from the dropped responses.
\[
\begin{aligned}
S &= \operatorname{TopK}_{m}\{1,\ldots,G\},\\
m_i &=
\begin{cases}
-L_i, & \text{shortest filter}\\
r_i/\max(L_i,1), & \text{token-efficiency filter}
\end{cases}\\[1mm]
\mu_S &= \frac{1}{|S|}\sum_{j\in S} r_j,\qquad
\sigma_S=\sqrt{\frac{1}{|S|}\sum_{j\in S}(r_j-\mu_S)^2}\\[1mm]
A_i^{\mathrm{GFPO}} &=
\mathbf{1}[i\in S]\frac{r_i-\mu_S}{\sigma_S+\epsilon}.
\end{aligned}
\]
Our reproduction uses $G=16$, $k=8$, raw rewards unchanged, and dropped
responses kept in batch with zero advantage.
\end{minipage}
""",
)

render(
    "eval_table",
    r"""
\begin{minipage}{8.8in}
\color{textdark}
\small
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{llll}
\toprule
\textbf{Domain} & \textbf{Benchmark} & \textbf{Primary metric} & \textbf{Purpose}\\
\midrule
Math, moderate & Minerva Math 500 (Lewkowycz et al., 2022) & exact-answer score & Core math reasoning\\
Math, hard & AIME 2025 (HF set) & pass@1 and pass@32 & Hard problems where lengths grow most\\
Code & LiveCodeBench generation (Jain et al., 2024) & pass@1 & Code generation and cross-domain brevity\\
Instruction following & IFEval / ifbench wildchat + OOD (Zhou et al., 2023) & instruction-following accuracy & Precise instruction following\\
Chat quality & AlpacaEval v3 (Li et al., 2023) & length-controlled win rate & General quality and verbosity\\
\bottomrule
\end{tabular}
\end{minipage}
""",
)

print("\n".join(str(p) for p in sorted(ASSET_DIR.glob("*.png"))))

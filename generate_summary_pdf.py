"""Generate 2-page 4th-year technical monograph PDF for Flux Visualiser v4.0."""

import os
from fpdf import FPDF


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HERO_IMG = os.path.join(BASE_DIR, "assets", "flux-visualiser-naca4412.png")
VAL_IMG = os.path.join(BASE_DIR, "assets", "flux-visualiser-validation.png")
OUT_PDF = os.path.join(BASE_DIR, "docs", "ProjectSummary_2D_CFD_JoanMartinezMiravet.pdf")


class MonographPDF(FPDF):
    def footer(self):
        self.set_y(-11)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(100, 116, 139)
        self.cell(
            0,
            6,
            f"Joan Martinez Miravet  |  Aerospace Systems Engineering (EETAC - UPC)  |  Flux Visualiser v4.0  |  Page {self.page_no()}/2",
            align="C",
        )


def draw_header(pdf: FPDF, title: str, subtitle: str) -> None:
    pdf.set_fill_color(12, 15, 20)
    pdf.rect(0, 0, 210, 25, "F")
    pdf.set_fill_color(56, 189, 248)
    pdf.rect(0, 24.2, 210, 0.8, "F")

    pdf.set_xy(12, 5.5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(248, 250, 252)
    pdf.cell(0, 6.5, title, ln=1)

    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 8.8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5.5, subtitle, ln=1)


def draw_section_heading(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 5.8, text, ln=1)
    y = pdf.get_y()
    pdf.set_draw_color(203, 213, 225)
    pdf.set_line_width(0.35)
    pdf.line(12, y, 198, y)
    pdf.ln(2.2)


def draw_card_row(pdf: FPDF, cards: list[tuple[str, str]], height: float = 21.0) -> None:
    y0 = pdf.get_y()
    col_w = 91.0
    gap = 4.0
    for idx, (heading, body) in enumerate(cards):
        x0 = 12 + idx * (col_w + gap)
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(x0, y0, col_w, height, "FD")
        pdf.set_xy(x0 + 2.5, y0 + 2.0)
        pdf.set_font("Helvetica", "B", 8.2)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(col_w - 5, 4.2, heading, ln=1)
        pdf.set_x(x0 + 2.5)
        pdf.set_font("Helvetica", "", 7.6)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(col_w - 5, 3.6, body)
    pdf.set_y(y0 + height + 3.0)


def build_pdf() -> None:
    pdf = MonographPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(False)

    # ------------------------------------------------------------------
    # PAGE 1: Architecture & High-Resolution CFD Dashboard
    # ------------------------------------------------------------------
    pdf.add_page()
    draw_header(
        pdf,
        "FLUX VISUALISER v4.0 // 2D COMPRESSIBLE CFD & VISCOUS SOLVER",
        "Joan Martinez Miravet  |  4th-Year Aerospace Systems Engineering Technical Monograph (EETAC - UPC)",
    )
    pdf.set_xy(12, 29)

    draw_section_heading(pdf, "1. Engineering Objective & Solver Evolution")
    pdf.set_font("Helvetica", "", 8.6)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(
        186,
        4.3,
        "Built from scratch in Python (NumPy, SciPy, OpenCV, Matplotlib) as a major upgrade from an iterative 2nd-year Laplace prototype "
        "into a validated 4th-year aerodynamic solver. The engine couples a 2nd-order O(dx^2) Shortley-Weller immersed-boundary Sparse LU "
        "Laplace solver with a vectorized 160-panel Hess-Smith source-vortex method, Prandtl-Glauert compressibility, and an integral viscous "
        "boundary-layer march (Thwaites -> Michel/separation transition -> Head entrainment -> Squire-Young profile drag).",
    )
    pdf.ln(2.5)

    draw_section_heading(pdf, "2. Four-Stage Computational Architecture")
    draw_card_row(
        pdf,
        [
            (
                "A. Immersed-Boundary Sparse LU (FDM)",
                "Non-uniform Shortley-Weller cut-cell discretization of Laplace's equation, SciPy SuperLU factorization, linear superposition "
                "(psi = psi_0 + C*psi_1), trailing-edge Kutta calibration, and wall-normal Taylor Cp probe.",
            ),
            (
                "B. Hess-Smith Panels + Prandtl-Glauert",
                "160 cosine-clustered panels with constant sources and uniform vorticity enforcing Vt[0] + Vt[N-1] = 0 Kutta condition and "
                "subsonic compressibility scaling beta = sqrt(1 - M^2).",
            ),
        ],
        height=23.5,
    )
    draw_card_row(
        pdf,
        [
            (
                "C. Coupled Viscous Boundary Layer",
                "Dual-branch stagnation march: Thwaites laminar momentum integral, Michel transition criterion, Head turbulent entrainment "
                "with Ludwieg-Tillmann Cf, and Squire-Young Cd.",
            ),
            (
                "D. Geometry Ingestion & Interactive UI",
                "Analytic NACA 4-digit generator (closed TE a4 = -0.1036), circular cylinder, Selig .txt parser, and OpenCV contour "
                "extraction from arbitrary PNG silhouettes plus live sliders.",
            ),
        ],
        height=21.5,
    )

    draw_section_heading(pdf, "3. Reference Simulation Dashboard (NACA 4412, alpha = 8.0 deg, U_inf = 60 m/s)")
    img_y = pdf.get_y()
    img_w = 186.0
    img_h = 107.0
    pdf.image(HERO_IMG, x=12, y=img_y, w=img_w, h=img_h)
    pdf.set_y(img_y + img_h + 2.5)

    # Telemetry strip below hero image
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(148, 163, 184)
    y_box = pdf.get_y()
    pdf.rect(12, y_box, 186, 17.5, "FD")
    pdf.set_xy(15, y_box + 2.2)
    pdf.set_font("Helvetica", "B", 8.0)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 4.2, "Reference Solution Telemetry (NACA 4412, c = 1 m, dx = 0.012 c -> 268x184 = 49,312 nodes):", ln=1)
    pdf.set_x(15)
    pdf.set_font("Courier", "", 6.3)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(
        0,
        4.2,
        "Ma = 0.176 (beta = 0.984) | Re_c = 4.05e6 | Cl (KJ) = 1.5032 | Cl (Cp) = 1.4908 | Cd_v = 0.00954 | L/D = 157.5",
        ln=1,
    )
    pdf.set_x(15)
    pdf.cell(
        0,
        4.2,
        "Cm(c/4) = -0.1228 | x_tr = 0.05 (up) / 0.96 (lo) | Wall-Cp RMSE = 0.193 | Sparse LU = 168.2 ms, Solve = 11.7 ms",
        ln=1,
    )

    # ------------------------------------------------------------------
    # PAGE 2: Verification & Validation (V&V) Quantitative Suite
    # ------------------------------------------------------------------
    pdf.add_page()
    draw_header(
        pdf,
        "VERIFICATION, VALIDATION & BENCHMARKS (V&V SUITE)",
        "Spatial Grid Convergence, Linear Algebra Speedup, Cross-Method Cp Validation & Viscous Polars",
    )
    pdf.set_xy(12, 29)

    draw_section_heading(pdf, "4. Quantitative V&V Evidence (2x2 Automated Test Suite)")
    vy = pdf.get_y()
    pdf.image(VAL_IMG, x=12, y=vy, w=186.0, h=121.8)
    pdf.set_y(vy + 125.0)

    draw_section_heading(pdf, "5. Benchmark Metrics & Engineering Findings")

    # Clean comparison table
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(248, 250, 252)
    pdf.set_font("Helvetica", "B", 7.8)
    widths = [26, 30, 32, 32, 34, 32]
    headers = ["Grid dx (c)", "Mesh Nodes", "Sparse LU (ms)", "Vec Jacobi (ms)", "Pure-Py Jacobi (ms)", "LU vs Py Speedup"]
    for w, h in zip(widths, headers):
        pdf.cell(w, 6.2, h, border=1, align="C", fill=True)
    pdf.ln()

    rows = [
        ["0.080", "41 x 29 (1,189)", "3.59 ms", "5.11 ms", "292.19 ms", "81.4x faster"],
        ["0.040", "81 x 56 (4,536)", "12.04 ms", "26.43 ms", "3,122.35 ms", "259.3x faster"],
        ["0.020", "161 x 111 (17,871)", "53.44 ms", "176.82 ms", "32,046.25 ms", "599.7x faster"],
    ]
    pdf.set_font("Helvetica", "", 7.8)
    for r_i, row in enumerate(rows):
        fill = r_i % 2 == 1
        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        for w, val in zip(widths, row):
            pdf.cell(w, 5.8, val, border=1, align="C", fill=fill)
        pdf.ln()

    pdf.ln(3.5)
    draw_card_row(
        pdf,
        [
            (
                "Analytical Grid Refinement (Cylinder)",
                "Normalized L2 error against exact dipole + vortex potential drops cleanly across dx = 0.08 -> 0.04 -> 0.02 -> 0.01: "
                "[6.47e-4, 1.60e-4, 4.17e-5, 1.07e-5], yielding a measured convergence order p = 1.974 (strict O(dx^2) accuracy).",
            ),
            (
                "Cross-Solver & Viscous Polar Findings",
                "On NACA 4412 (alpha = 8 deg, dx = 0.010), wall-normal Taylor extraction matches Hess-Smith surface Cp with RMSE = 0.154. "
                "Attached-flow polars sweep alpha in [-4, 12] deg smoothly from zero-alpha viscous drag (Cd_v = 0.00518) to LE transition.",
            ),
        ],
        height=27.0,
    )

    pdf.ln(1.5)
    pdf.set_font("Helvetica", "B", 8.2)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 4.5, "Modeling Scope & Engineering Assumptions:", ln=1)
    pdf.set_font("Helvetica", "", 7.8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(
        186,
        3.9,
        "Intended for subsonic attached-flow regimes up to trailing-edge separation onset (M < 0.7). Compressibility uses subsonic Prandtl-Glauert "
        "scaling without shock capturing, and post-stall separated wakes are outside attached integral boundary-layer validity. Reproducible "
        "via CLI: `python flux_visualaiser.py --export-figures --out-dir ./assets`.",
    )

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    pdf.output(OUT_PDF)
    print(f"Wrote PDF: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()

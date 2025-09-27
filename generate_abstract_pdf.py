import os

from fpdf import FPDF


ABSTRACT_TITLE = "Driver Drowsiness Detection - Project Abstract"

ABSTRACT_TEXT = (
    "This project implements a real-time Driver Drowsiness Detection system using a standard "
    "webcam, OpenCV, and dlib facial landmarks. The system continuously detects the driver's "
    "face and extracts eye and mouth landmarks to compute Eye Aspect Ratio (EAR) and Mouth "
    "Aspect Ratio (MAR). A smoothed, time-based EAR criterion determines drowsiness: if EAR "
    "remains below a calibrated threshold for a configurable duration (e.g., 7-10 seconds), "
    "the system flags 'Drowsy' and triggers an audible alarm.\n\n"
    "Robustness is improved via largest-face selection, optional auto-calibration of the EAR "
    "threshold, minimum face size filtering, FPS-aware overlays, detector interval reuse and "
    "best-face selection for efficiency, and Windows-friendly camera backend fallback. Audio "
    "behavior includes an immediate-stop mechanism with a release margin to prevent lingering "
    "alarms and an optional brief 'awake' chime when recovery is detected. The application is "
    "configurable via CLI and provides visual feedback (status, EAR/MAR, FPS).\n\n"
    "This implementation is intended for demonstration, prototyping, and educational use, and "
    "is not a medical or safety-certified system."
)

KEYWORDS = (
    "Driver monitoring; drowsiness detection; EAR; MAR; OpenCV; dlib; real-time; webcam"
)


def build_pdf(output_path: str = "ABSTRACT.pdf") -> str:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.multi_cell(0, 10, ABSTRACT_TITLE)
    pdf.ln(2)

    # Body
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, ABSTRACT_TEXT)
    pdf.ln(2)

    # Keywords
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 7, "Keywords:")
    pdf.ln(7)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, KEYWORDS)

    pdf.output(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    path = build_pdf()
    print(f"[INFO] Wrote abstract PDF to: {path}")



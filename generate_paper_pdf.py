import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import mm

OUTPUT_PATH = os.path.join("paper", "IEEE_Paper.pdf")


def build_pdf(output_path: str = OUTPUT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Real-Time Driver Drowsiness Detection using Eye Aspect Ratio and Facial Landmarks",
        author="Mohan [Surname]",
    )

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    body_style = styles['BodyText']

    story = []

    # Title
    story.append(Paragraph("Real-Time Driver Drowsiness Detection using Eye Aspect Ratio and Facial Landmarks", title_style))
    story.append(Paragraph("Mohan [Surname]", body_style))
    story.append(Spacer(1, 6))

    # Abstract
    story.append(Paragraph("Abstract", heading_style))
    story.append(Paragraph(
        "This paper presents a practical, real-time driver drowsiness detection system using a standard webcam, OpenCV, and the dlib 68-point facial landmark model. "
        "The system estimates Eye Aspect Ratio (EAR) and optionally Mouth Aspect Ratio (MAR) to detect sustained eye closure and potential yawning. To minimize false alarms, "
        "it uses exponential moving average smoothing, a time-based EAR threshold, release margins, and alarm hysteresis. The system runs on CPU-only hardware, provides clear "
        "visual overlays and audio alerts (including an immediate stop with a brief 'awake' chime), and includes a Windows-friendly camera backend fallback. Experiments on a laptop "
        "webcam demonstrate real-time performance and robust behavior under typical lighting and head pose conditions. While not safety-certified, the design is well-suited for "
        "demonstrations, education, and prototyping of driver monitoring features.",
        body_style))
    story.append(Spacer(1, 6))

    # Keywords
    story.append(Paragraph("Keywords", heading_style))
    story.append(Paragraph("Drowsiness detection; EAR; MAR; facial landmarks; OpenCV; dlib; real-time; driver monitoring", body_style))
    story.append(Spacer(1, 6))

    # Sections
    story.append(Paragraph("1. Introduction", heading_style))
    story.append(Paragraph(
        "Drowsy driving remains a significant safety concern, motivating in-cabin monitoring that can detect fatigue in real time. We implement a low-cost system relying on a "
        "standard webcam and CPU, using facial landmarks to compute EAR and MAR as interpretable indicators. The approach emphasizes configurability and ease of deployment while "
        "maintaining real-time performance.", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("2. Related Work", heading_style))
    story.append(Paragraph(
        "Facial landmarks for blink and eye-closure estimation are widely used baselines. PERCLOS and EAR offer lightweight heuristics. Dlib's regression-tree landmark model with "
        "HOG-based detection is common, and OpenCV provides real-time video primitives.", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("3. Methodology", heading_style))
    story.append(Paragraph(
        "We compute EAR from six eye landmarks per eye and average across both eyes. To reduce false positives, we smooth EAR via an exponential moving average and apply a time-based "
        "threshold (e.g., 7-10 seconds below threshold signals drowsiness). Recovery is enforced with a release margin and short dwell time to stop the alarm when clearly awake. MAR "
        "serves as an optional secondary cue for yawning.", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("4. System Design and Implementation", heading_style))
    story.append(Paragraph(
        "The pipeline uses OpenCV for capture/display, dlib HOG detector and 68-point predictor for face/landmarks, and largest-face selection. To improve efficiency, detections are reused "
        "between frames; on Windows, MSMF to DSHOW to ANY backend fallback is provided. Overlays show status, EAR/MAR, and FPS. Audio uses a looping alarm and a brief 'awake' chime on recovery "
        "with immediate-stop control.", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("5. Experiments and Results", heading_style))
    story.append(Paragraph(
        "On a laptop (CPU-only) with a 720p webcam, the system maintained real-time performance (>=15-20 FPS) and reliably detected sustained eye closure. Parameters (thresholds, durations, smoothing, "
        "margin) were tuned for stability, and an optional auto-calibration established a personalized baseline.", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("6. Conclusion and Future Work", heading_style))
    story.append(Paragraph(
        "We demonstrated a practical, interpretable, and configurable drowsiness detector running in real time on consumer hardware. Future work includes lightweight learning-based classifiers for tougher conditions, "
        "multi-session personalization, head-nod/blink-pattern features, and packaging as a desktop app with logs.", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("References", heading_style))
    refs = [
        "[1] NHTSA. Drowsy Driving Research and Program Plan, 2023.",
        "[2] T. Soukupova, J. Cech. Real-Time Eye Blink Detection Using Facial Landmarks, CVWW 2016.",
        "[3] V. Kazemi, J. Sullivan. One Millisecond Face Alignment with an Ensemble of Regression Trees, CVPR 2014.",
        "[4] N. Dalal, B. Triggs. Histograms of Oriented Gradients for Human Detection, CVPR 2005.",
        "[5] OpenCV Library, https://opencv.org (accessed 2025-08).",
        "[6] W. Wierwille, L. Ellsworth. Evaluation of Driver Drowsiness by Eye Movement Measures and PERCLOS, SAE 1994."
    ]
    for r in refs:
        story.append(Paragraph(r, body_style))

    doc.build(story)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    path = build_pdf()
    print(f"[INFO] Wrote paper PDF to: {path}")



from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime


def generate_report(name, age, sex, bpm, breathing, risk):

    file_name = f"report_{name}.pdf"
    doc = SimpleDocTemplate(file_name)

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph("<b>Cardiac Monitoring System</b>", styles["Title"]))
    content.append(
        Paragraph("We help you prevent it before it's too late", styles["Normal"]))
    content.append(Spacer(1, 20))

    patient = [
        ["Name", name],
        ["Age", age],
        ["Sex", sex],
        ["Time", str(datetime.now())]
    ]

    t1 = Table(patient)
    t1.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    content.append(t1)

    content.append(Spacer(1, 20))

    vitals = [
        ["Heart Rate", bpm],
        ["Breathing", breathing],
        ["Risk", risk]
    ]

    t2 = Table(vitals)
    t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    content.append(t2)

    content.append(Spacer(1, 30))
    content.append(
        Paragraph("Doctor Signature: ____________", styles["Normal"]))

    doc.build(content)

    return file_name

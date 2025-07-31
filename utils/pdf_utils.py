import os
import logging
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def generate_pdf(results, subject_name, features_df, inputs_used, output_dir='temp'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f'report_{subject_name}.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Reporte de Predicción de Demencia", styles['Title']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph(f"Nombre del Sujeto: {subject_name}", styles['Heading2']))
        story.append(Paragraph(f"Edad: {inputs_used['age']}", styles['Normal']))
        story.append(Paragraph(f"Género: {inputs_used['gender']}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Resultados de Predicción", styles['Heading2']))
        data = [
            ['Categoría', 'Valor'],
            ['Grupo de Demencia', results['group']],
            ['Puntaje MMSE Estimado', f"{results['mmse']:.2f}"],
            ['Intervalo de Confianza MMSE', results['mmse_ic']]
        ]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Métricas de Rendimiento (Bootstrap)", styles['Heading2']))
        metrics_data = [['Métrica', 'Valor']] + [[k, v] for k, v in results['metrics'].items()]
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Visualización de Features Espectrales", styles['Heading2']))
        fig, ax = plt.subplots()
        ax.bar(BANDS.keys(), [features_df.filter(like=f'abs_power_{band}_').mean().mean() for band in BANDS])
        ax.set_title('Power Absoluto Promedio por Banda')
        ax.set_ylabel('Power')
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image(buf, width=4*inch, height=3*inch)
        story.append(img)
        plt.close(fig)

        doc.build(story)
        logging.info(f"PDF generado: {pdf_path}")
        return pdf_path

    except Exception as e:
        logging.error(f"Error generando PDF: {e}")
        raise

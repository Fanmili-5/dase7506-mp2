"""Render the editable report manuscript; no model or data is loaded."""
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor('#243442')
ACCENT = colors.HexColor('#276278')
PALE = colors.HexColor('#F0F5F7')


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',
                  r'<link href="\2" color="#276278">\1</link>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)


def main():
    source = ROOT / 'report/MP2_Report.md'
    manuscript = source.read_text()
    test = json.loads((ROOT / 'evaluation/test-score.json').read_text())
    assert test['correct'] == 399 and test['n'] == 467
    output = ROOT / 'report/MP2_Report.pdf'
    doc = SimpleDocTemplate(
        str(output), pagesize=(210 * mm, 297 * mm),
        leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=18 * mm, bottomMargin=19 * mm,
        title='Choosing among four mathematical solutions - DASE7506 MP2',
        author='', subject='Relative and joint verifiers; test 399/467',
    )
    styles = {
        'title': ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=23,
                                leading=27, textColor=INK, spaceAfter=12),
        'subtitle': ParagraphStyle('Subtitle', fontName='Helvetica', fontSize=10,
                                   leading=14, textColor=ACCENT, spaceAfter=15),
        'section': ParagraphStyle('Section', fontName='Helvetica-Bold', fontSize=14,
                                  leading=18, textColor=INK, spaceBefore=7, spaceAfter=9),
        'subsection': ParagraphStyle('Subsection', fontName='Helvetica-Bold', fontSize=10.4,
                                     leading=14, textColor=INK, spaceBefore=9, spaceAfter=5),
        'body': ParagraphStyle('Body', fontName='Helvetica', fontSize=9.5,
                               leading=13.2, spaceAfter=8),
        'reference': ParagraphStyle('Reference', fontName='Helvetica', fontSize=8.1,
                                    leading=10.8, spaceAfter=6),
        'cell': ParagraphStyle('Cell', fontName='Helvetica', fontSize=8.5, leading=11),
        'code': ParagraphStyle('Code', fontName='Courier', fontSize=8.2,
                               leading=11, backColor=PALE, borderPadding=6,
                               spaceBefore=4, spaceAfter=10),
    }
    story = []
    lines = manuscript.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith('```'):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(escape(lines[i]).replace(' ', '&nbsp;'))
                i += 1
            story.append(Paragraph('<br/>'.join(block), styles['code']))
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-\s]+', s) for s in row):
                    rows.append(row)
                i += 1
            table_widths = {
                'Split': [0.17, 0.43, 0.40],
                'Method': [0.60, 0.20, 0.20],
                'Fixed evaluated version': [0.60, 0.20, 0.20],
                'Measurement': [0.45, 0.35, 0.20],
                'Recent experiment': [0.49, 0.17, 0.34],
            }
            widths = table_widths.get(rows[0][0], [0.55, 0.45])
            cells = [[Paragraph(inline(s), styles['cell']) for s in row] for row in rows]
            table = Table(cells, colWidths=[doc.width * w for w in widths],
                          repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PALE),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LINEBELOW', (0, 0), (-1, 0), .7, ACCENT),
                ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#D5DFE4')),
            ]))
            story.extend([table, Spacer(1, 8)])
            continue
        if line.startswith('# '):
            style, text = 'title', line[2:]
        elif line.startswith('## '):
            style, text = 'section', line[3:]
        elif line.startswith('### '):
            style, text = 'subsection', line[4:]
        else:
            style = 'reference' if re.match(r'^\[\d+\] ', line) else 'body'
            if line.startswith('DASE7506 Mini-Project'):
                style = 'subtitle'
            paragraph = [line]
            while i + 1 < len(lines) and lines[i + 1].strip():
                i += 1
                paragraph.append(lines[i].strip())
            text = ' '.join(paragraph)
        story.append(Paragraph(inline(text), styles[style]))
        i += 1

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(.4)
        canvas.line(19 * mm, 14 * mm, 191 * mm, 14 * mm)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(INK)
        canvas.drawString(19 * mm, 10 * mm, 'DASE7506 MP2 | Choosing reliable answers')
        canvas.drawRightString(191 * mm, 10 * mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    assert doc.page <= 10
    (ROOT / 'report/MP2_Report_source.txt').write_text(manuscript)
    print(json.dumps({'report': str(output), 'pages': doc.page}))


if __name__ == '__main__':
    main()

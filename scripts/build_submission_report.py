"""Compile the LaTeX report with Tectonic; model assets are not loaded."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    from pypdf import PdfReader

    engine = shutil.which('tectonic')
    if engine is None:
        raise SystemExit('Install Tectonic, or compile report/MP2_Report.tex on Overleaf.')
    source = ROOT / 'report/MP2_Report.tex'
    with tempfile.TemporaryDirectory(prefix='mp2-report-') as build:
        subprocess.run([engine, '--keep-logs', '--outdir', build, str(source)], check=True)
        pdf = Path(build) / 'MP2_Report.pdf'
        pages = len(PdfReader(pdf).pages)
        if not 1 <= pages <= 10:
            raise RuntimeError(f'Report has {pages} pages; the limit is ten.')
        destination = ROOT / 'report/MP2_Report.pdf'
        shutil.copyfile(pdf, destination)
    record = {
        'format': 'LaTeX',
        'source': 'report/MP2_Report.tex',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'bytes': destination.stat().st_size,
        'pages': pages,
        'compiler': subprocess.check_output([engine, '--version'], text=True).strip(),
    }
    (ROOT / 'report/REPORT_BUILD.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

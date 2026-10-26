"""
Export Reports to Print-Ready HTML and PDF
UAC Program Care Load & Placement Demand Forecasting
"""

import os
import subprocess
import markdown

CSS_STYLE = """
<style>
    @page {
        size: A4;
        margin: 18mm 18mm 20mm 18mm;
        @bottom-right {
            content: counter(page);
        }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.6;
        color: #1e293b;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }
    h1 {
        font-size: 20pt;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #0284c7;
        padding-bottom: 8px;
        margin-top: 0;
        margin-bottom: 16px;
        page-break-after: avoid;
    }
    h2 {
        font-size: 14pt;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 24px;
        margin-bottom: 12px;
        page-break-after: avoid;
    }
    h3 {
        font-size: 12pt;
        font-weight: 600;
        color: #334155;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }
    p {
        margin-top: 0;
        margin-bottom: 12px;
        text-align: justify;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }
    th, td {
        padding: 8px 10px;
        text-align: left;
        border: 1px solid #cbd5e1;
    }
    th {
        background-color: #f1f5f9;
        font-weight: 600;
        color: #0f172a;
    }
    tr:nth-child(even) td {
        background-color: #f8fafc;
    }
    blockquote {
        margin: 14px 0;
        padding: 10px 16px;
        background-color: #f0f9ff;
        border-left: 4px solid #0284c7;
        color: #0369a1;
        font-style: normal;
        border-radius: 4px;
        page-break-inside: avoid;
    }
    blockquote p {
        margin: 0;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 16px auto 6px auto;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        page-break-inside: avoid;
    }
    em {
        display: block;
        text-align: center;
        font-size: 9pt;
        color: #64748b;
        margin-bottom: 16px;
    }
    code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 9pt;
        background-color: #f1f5f9;
        padding: 2px 5px;
        border-radius: 4px;
        color: #0f172a;
    }
    pre {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 12px 16px;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 9pt;
        page-break-inside: avoid;
    }
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
    }
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 24px 0;
    }
    .badge-green { color: #16a34a; font-weight: bold; }
    .badge-yellow { color: #ca8a04; font-weight: bold; }
    .badge-orange { color: #ea580c; font-weight: bold; }
    .badge-red { color: #dc2626; font-weight: bold; }
</style>
"""

def convert_md_to_html(md_path: str, html_path: str, title: str):
    """Converts markdown to standalone HTML with embedded styling."""
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Convert to HTML with extensions
    html_content = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'toc', 'sane_lists']
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {CSS_STYLE}
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"Generated HTML: {html_path}")

def convert_html_to_pdf(html_path: str, pdf_path: str):
    """Uses Google Chrome headless printing to generate high-resolution PDF."""
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)
    cmd = [
        'google-chrome',
        '--headless=new',
        '--disable-gpu',
        '--no-pdf-header-footer',
        f'--print-to-pdf={abs_pdf}',
        f'file://{abs_html}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(abs_pdf):
        print(f"Successfully generated PDF: {abs_pdf} ({os.path.getsize(abs_pdf):,} bytes)")
    else:
        print(f"Failed to generate PDF. Returncode: {result.returncode}, Error: {result.stderr}")

if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    
    # 1. Executive Summary
    exec_md = 'reports/executive_summary.md'
    exec_html = 'reports/executive_summary.html'
    exec_pdf = 'reports/executive_summary.pdf'
    convert_md_to_html(exec_md, exec_html, "Executive Summary: Predictive Care Load & Placement Intelligence")
    convert_html_to_pdf(exec_html, exec_pdf)

    # 2. Research Paper
    paper_md = 'reports/research_paper.md'
    paper_html = 'reports/research_paper.html'
    paper_pdf = 'reports/research_paper.pdf'
    convert_md_to_html(paper_md, paper_html, "Research Paper: Predictive Forecasting of Care Load & Placement Demand")
    convert_html_to_pdf(paper_html, paper_pdf)

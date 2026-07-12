import os
import markdown
from xhtml2pdf import pisa

def compile_pdf(md_path, pdf_path):
    print(f"Reading markdown from: {md_path}")
    if not os.path.exists(md_path):
        print(f"Error: {md_path} does not exist.")
        return False
        
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to html with table and code highlighting support
    html_body = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'fenced_code']
    )

    # Balanced custom styling to target exactly 8 pages
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: letter;
                margin-top: 0.75in;
                margin-bottom: 0.75in;
                margin-left: 0.75in;
                margin-right: 0.75in;
                @frame header {{
                    -pdf-frame-content: header-content;
                    top: 0.35in;
                    left: 0.75in;
                    width: 7.0in;
                    height: 0.3in;
                }}
                @frame footer {{
                    -pdf-frame-content: footer-content;
                    bottom: 0.35in;
                    left: 0.75in;
                    width: 7.0in;
                    height: 0.3in;
                }}
            }}
            
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 9.0pt;
                line-height: 1.25;
                color: #222222;
            }}
            
            h1 {{
                font-size: 15pt;
                text-align: center;
                margin-top: 0pt;
                margin-bottom: 4pt;
                color: #111111;
                font-weight: bold;
            }}
            
            h2 {{
                font-size: 10.5pt;
                margin-top: 10pt;
                margin-bottom: 4pt;
                color: #113366;
                font-weight: bold;
                border-bottom: 0.5pt solid #336699;
                padding-bottom: 2pt;
            }}
            
            h3 {{
                font-size: 9.5pt;
                margin-top: 8pt;
                margin-bottom: 3pt;
                color: #222222;
                font-weight: bold;
            }}
            
            h4 {{
                font-size: 9.0pt;
                margin-top: 6pt;
                margin-bottom: 2pt;
                color: #444444;
                font-weight: bold;
            }}
            
            p {{
                margin-bottom: 4pt;
                text-align: justify;
            }}
            
            ul, ol {{
                margin-bottom: 4pt;
                padding-left: 12pt;
            }}
            
            li {{
                margin-bottom: 2pt;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 6pt;
                margin-bottom: 6pt;
            }}
            
            th {{
                background-color: #f2f5f9;
                font-weight: bold;
                border: 0.5pt solid #aaaaaa;
                padding: 3pt;
                font-size: 8.0pt;
                color: #113366;
            }}
            
            td {{
                border: 0.5pt solid #cccccc;
                padding: 3pt;
                font-size: 8.0pt;
            }}
            
            pre {{
                font-family: Courier, monospace;
                font-size: 7.5pt;
                background-color: #f7f9fa;
                border-left: 2.0pt solid #336699;
                padding: 4pt;
                margin-top: 4pt;
                margin-bottom: 4pt;
                white-space: pre-wrap;
            }}
            
            code {{
                font-family: Courier, monospace;
                font-size: 8.0pt;
                background-color: #f4f6f8;
                padding: 0.5pt 1.5pt;
            }}
            
            .center-align {{
                text-align: center;
            }}
            
            .subtitle {{
                font-size: 10pt;
                text-align: center;
                margin-bottom: 12pt;
                color: #555555;
                font-style: italic;
            }}
            
            hr {{
                border: 0;
                border-top: 0.5pt solid #dddddd;
                margin-top: 10pt;
                margin-bottom: 10pt;
            }}
            
            #header-content {{
                font-size: 7.5pt;
                color: #777777;
                text-align: left;
                border-bottom: 0.5pt solid #dddddd;
                padding-bottom: 2pt;
            }}
            
            #footer-content {{
                font-size: 7.5pt;
                color: #777777;
                text-align: right;
                border-top: 0.5pt solid #dddddd;
                padding-top: 2pt;
            }}
        </style>
    </head>
    <body>
        <!-- Header & Footer Content for xhtml2pdf frame injection -->
        <div id="header-content">
            Satellite Image Land-Use Classifier & Temporal Change Detection Report
        </div>
        <div id="footer-content">
            Page <pdf:pagenumber/> of <pdf:pagecount/>
        </div>
        
        <!-- Main Report Body -->
        {html_body}
    </body>
    </html>
    """

    print(f"Compiling PDF to: {pdf_path}")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(
            html_template,
            dest=pdf_file
        )
        
    if pisa_status.err:
        print(f"Error occurred during PDF generation: {pisa_status.err}")
        return False
        
    print("PDF compilation successful!")
    return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_file = os.path.join(base_dir, "report", "REPORT.md")
    
    # Compile to project root
    pdf_file_root = os.path.join(base_dir, "REPORT.pdf")
    compile_pdf(md_file, pdf_file_root)
    
    # Compile to outputs/reports/
    pdf_file_outputs = os.path.join(base_dir, "outputs", "reports", "REPORT.pdf")
    compile_pdf(md_file, pdf_file_outputs)

from pypdf import PdfReader

pdf_path = r"data\documents\lt-valves-gate-globe-check-valves-asme-b1634.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages):

    text = page.extract_text()

    print(f"\n--- PAGE {page_number + 1} ---\n")

    print(text[:1000])
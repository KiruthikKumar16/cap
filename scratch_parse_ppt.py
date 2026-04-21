import zipfile
import xml.etree.ElementTree as ET

namespaces = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
}

def extract_text_from_pptx(pptx_path):
    with zipfile.ZipFile(pptx_path, 'r') as z:
        for filename in z.namelist():
            if filename.startswith('ppt/slides/slide'):
                xml_content = z.read(filename)
                root = ET.fromstring(xml_content)
                print(f"--- {filename} ---")
                for sp in root.findall('.//p:sp', namespaces):
                    text_parts = []
                    for t in sp.findall('.//a:t', namespaces):
                        if t.text:
                            text_parts.append(t.text)
                    text = " ".join(text_parts)
                    words = text.split()
                    if words:
                        print(f"Word count: {len(words)} | Prefix: {' '.join(words[:10])}...")

extract_text_from_pptx('posterTemplate (1).pptx')

import logfire
from unstructured.partition.auto import partition

def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using the Unstructured library.
    Unlike PDFs, these formats are structured and lightweight, so they are processed locally.
    """
    with logfire.span("📄 Office Document Parsing", filename=file_path):
        lower_path = file_path.lower()
        try:
            if lower_path.endswith(".docx"):
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = "\n".join([p.text for p in doc.paragraphs if p.text])
                    logfire.info(f"✅ Successfully parsed {len(text)} characters via python-docx")
                    return text
                except Exception as docx_err:
                    logfire.warning(f"⚠️ python-docx failed ({docx_err}), falling back to unstructured")
            elif lower_path.endswith(".pptx"):
                try:
                    from pptx import Presentation
                    prs = Presentation(file_path)
                    text_runs = []
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text") and shape.text:
                                text_runs.append(shape.text)
                    text = "\n".join(text_runs)
                    logfire.info(f"✅ Successfully parsed {len(text)} characters via python-pptx")
                    return text
                except Exception as pptx_err:
                    logfire.warning(f"⚠️ python-pptx failed ({pptx_err}), falling back to unstructured")

            # Fallback to Unstructured library
            elements = partition(filename=file_path)
            full_text = "\n".join([str(el) for el in elements])
            
            if not full_text.strip():
                logfire.warning(f"⚠️ Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"✅ Successfully parsed {len(full_text)} characters via unstructured")

            return full_text
        except Exception as e:
            logfire.error(f"❌ Office Parse Failed: {e}")
            raise e
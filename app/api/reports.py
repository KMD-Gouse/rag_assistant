"""Read-only, manifest-scoped access to original report pages."""

from pathlib import Path

import pymupdf
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.ingestion.manifest import MANIFEST_PATH, PROJECT_ROOT, load_manifest


def report_router(manifest_path: Path = MANIFEST_PATH, reports_dir: Path = PROJECT_ROOT / "data" / "raw"):
    router = APIRouter(prefix="/reports")

    def resolve(doc_id):
        try:
            metadata = next((doc for doc in load_manifest(manifest_path).documents if doc.doc_id == doc_id), None)
        except (OSError, ValueError) as error:
            raise HTTPException(503, "Report catalog is unavailable.") from error
        if metadata is None:
            raise HTTPException(404, "Report not found.")
        root = reports_dir.resolve()
        path = (root / metadata.filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "Report PDF is unavailable.")
        return metadata, path

    def open_pdf(path):
        try:
            # Reading bytes closes the OS handle even when MuPDF rejects a corrupt file.
            pdf = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
            if not pdf.is_pdf or pdf.needs_pass:
                pdf.close()
                raise ValueError("Unreadable PDF")
            return pdf
        except (RuntimeError, ValueError, OSError) as error:
            raise HTTPException(503, "Report PDF could not be opened.") from error

    @router.get("/{doc_id}")
    def metadata(doc_id: str):
        report, path = resolve(doc_id)
        with open_pdf(path) as pdf:
            return {"doc_id": report.doc_id, "title": report.title, "pdf_pages": len(pdf)}

    @router.get("/{doc_id}/pdf")
    def download(doc_id: str):
        report, path = resolve(doc_id)
        return FileResponse(path, media_type="application/pdf", filename=Path(report.filename).name)

    @router.get("/{doc_id}/pages/{page}")
    def page_image(doc_id: str, page: int):
        _, path = resolve(doc_id)
        with open_pdf(path) as pdf:
            if not 1 <= page <= len(pdf):
                raise HTTPException(404, "PDF page not found.")
            # Citations use one-based PDF pages, independent of printed labels.
            image = pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            return Response(image.tobytes("png"), media_type="image/png")

    return router

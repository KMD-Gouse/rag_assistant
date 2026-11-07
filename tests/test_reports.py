import json

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.reports import report_router
from app.ingestion.manifest import load_manifest


@pytest.fixture
def reports(tmp_path):
    document = load_manifest().documents[0].model_dump()
    document.update(doc_id="sample", filename="sample.pdf", pdf_pages=2)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "documents": [document]}))
    with pymupdf.open() as pdf:
        for color in [(1, 0, 0), (0, 0, 1)]:
            page = pdf.new_page(width=100, height=100)
            page.draw_rect(page.rect, color=color, fill=color)
        pdf.save(tmp_path / "sample.pdf")
    app = FastAPI()
    app.include_router(report_router(manifest, tmp_path))
    return TestClient(app), tmp_path, manifest


def test_original_pdf_and_exact_one_based_page(reports):
    client, root, _ = reports
    assert client.get("/reports/sample").json()["pdf_pages"] == 2
    response = client.get("/reports/sample/pdf")
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == (root / "sample.pdf").read_bytes()
    for page, expected in [(1, (255, 0, 0)), (2, (0, 0, 255))]:
        response = client.get(f"/reports/sample/pages/{page}")
        assert response.headers["content-type"] == "image/png"
        assert pymupdf.Pixmap(response.content).pixel(50, 50) == expected


@pytest.mark.parametrize("path", ["unknown", "unknown/pdf", "unknown/pages/1", "sample/pages/0", "sample/pages/3"])
def test_unknown_reports_and_pages(reports, path):
    assert reports[0].get(f"/reports/{path}").status_code == 404


def test_missing_and_unreadable_pdf(reports):
    client, root, _ = reports
    (root / "sample.pdf").write_bytes(b"not a PDF")
    assert client.get("/reports/sample/pages/1").status_code == 503
    (root / "sample.pdf").unlink()
    assert client.get("/reports/sample").status_code == 404


def test_manifest_cannot_serve_files_outside_report_directory(reports):
    client, _, manifest = reports
    data = json.loads(manifest.read_text())
    data["documents"][0]["filename"] = "../outside.pdf"
    manifest.write_text(json.dumps(data))
    assert client.get("/reports/sample/pdf").status_code == 404

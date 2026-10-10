import uuid
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .services import ingest_pdf, search_pdf


MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB


@require_http_methods(["GET", "POST"])
def home(request):
    results = []
    document_id = request.session.get("document_id")
    filename = request.session.get("filename")
    chunk_count = request.session.get("chunk_count")
    page_count = request.session.get("page_count")

    if request.method == "POST":
        action = request.POST.get("action")

        # -------------------------
        # Upload and process PDF
        # -------------------------
        if action == "upload":
            uploaded_file = request.FILES.get("pdf_file")

            if not uploaded_file:
                messages.error(
                    request,
                    "Please select a PDF file."
                )

            elif Path(uploaded_file.name).suffix.lower() != ".pdf":
                messages.error(
                    request,
                    "Only PDF files are allowed."
                )

            elif uploaded_file.size > MAX_PDF_SIZE:
                messages.error(
                    request,
                    "The PDF must be 10 MB or smaller."
                )

            elif uploaded_file.size == 0:
                messages.error(
                    request,
                    "The uploaded file is empty."
                )

            else:
                saved_name = None

                try:
                    # Generate an ID for this specific upload.
                    new_document_id = uuid.uuid4().hex

                    # Save the uploaded file inside media/uploads.
                    saved_name = default_storage.save(
                        f"uploads/{new_document_id}.pdf",
                        uploaded_file,
                    )

                    pdf_path = default_storage.path(saved_name)

                    stats = ingest_pdf(
                        pdf_path=pdf_path,
                        document_id=new_document_id,
                        filename=Path(uploaded_file.name).name,
                    )

                    # Store the active PDF information in session.
                    request.session["document_id"] = new_document_id
                    request.session["filename"] = Path(
                        uploaded_file.name
                    ).name
                    request.session["chunk_count"] = stats["chunks"]
                    request.session["page_count"] = stats["pages"]

                    # Clear old search results after a new upload.
                    request.session.pop("last_query", None)

                    messages.success(
                        request,
                        (
                            f"PDF processed successfully! "
                            f"{stats['pages']} pages and "
                            f"{stats['chunks']} chunks indexed."
                        ),
                    )

                    document_id = new_document_id
                    filename = Path(uploaded_file.name).name
                    chunk_count = stats["chunks"]
                    page_count = stats["pages"]

                except Exception as exc:
                    # Remove the saved PDF if ingestion fails.
                    if saved_name:
                        default_storage.delete(saved_name)

                    messages.error(
                        request,
                        f"Could not process this PDF: {exc}"
                    )

        # -------------------------
        # Search the active PDF
        # -------------------------
        elif action == "search":
            query = request.POST.get("query", "").strip()

            if not document_id:
                messages.error(
                    request,
                    "Upload and process a PDF before searching."
                )

            elif not query:
                messages.error(
                    request,
                    "Please enter a search question."
                )

            elif len(query) > 1000:
                messages.error(
                    request,
                    "Your search query must be 1000 characters or less."
                )

            else:
                try:
                    results = search_pdf(
                        query=query,
                        document_id=document_id,
                        top_k=2,
                    )

                    request.session["last_query"] = query

                    if not results:
                        messages.info(
                            request,
                            "No searchable chunks were found."
                        )

                except Exception as exc:
                    messages.error(
                        request,
                        f"Search failed: {exc}"
                    )

    return render(
        request,
        "assistant/home.html",
        {
            "filename": filename,
            "chunk_count": chunk_count,
            "page_count": page_count,
            "results": results,
            "last_query": request.session.get("last_query", ""),
        },
    )
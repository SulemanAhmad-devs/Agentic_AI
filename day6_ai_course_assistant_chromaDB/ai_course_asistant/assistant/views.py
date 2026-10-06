from django.shortcuts import render

from .chroma_setup import search_course


def home(request):

    results = []
    question = ""

    if request.method == "POST":

        question = request.POST.get(
            "question",
            ""
        ).strip()

        if question:

            chroma_results = search_course(question)

            documents = chroma_results["documents"][0]
            metadatas = chroma_results["metadatas"][0]
            distances = chroma_results["distances"][0]

            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances
            ):

                results.append({
                    "document": document,
                    "category": metadata["category"],
                    "distance": round(distance, 4),
                })

    return render(
        request,
        "assistant/index.html",
        {
            "results": results,
            "question": question,
        }
    )
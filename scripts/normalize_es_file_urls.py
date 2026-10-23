import argparse
import json
import urllib.request
import urllib.error


def request_json(method, url, body=None):
    data = None
    headers = {}

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        print("Erro HTTP:", e.code)
        print(e.read().decode("utf-8"))
        raise


def request_ndjson(method, url, ndjson_text):
    data = ndjson_text.encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/x-ndjson"},
        method=method,
    )

    try:
        with urllib.request.urlopen(req) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        print("Erro HTTP:", e.code)
        print(e.read().decode("utf-8"))
        raise


def normalize_file_url(file_url):
    if not isinstance(file_url, str):
        return file_url

    path = file_url.strip()
    path = path.replace("\\", "/")

    markers = [
        "/src/main/java/",
        "/src/test/java/",
        "/src/java/",
        "/src/test/",
        "/src/experimental/",
        "/src/site/resources/userguide/",
        "/src/userguide/java/",
    ]

    for marker in markers:
        if marker in path:
            path = path.split(marker, 1)[1]
            path = path.strip("/")
            path = path.replace("/", ".")
            return path

    return file_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="9200")
    parser.add_argument("--index", required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    index_url = f"{base_url}/{args.index}"

    search_body = {
        "size": args.batch_size,
        "_source": ["file_url", "file_url_original"],
        "query": {
            "exists": {
                "field": "file_url"
            }
        }
    }

    print(f"Lendo documentos do índice: {args.index}")

    first_response = request_json(
        "POST",
        f"{index_url}/_search?scroll=2m",
        search_body,
    )

    scroll_id = first_response.get("_scroll_id")
    hits = first_response.get("hits", {}).get("hits", [])

    total_seen = 0
    total_changed = 0
    total_updated = 0

    examples = []

    while hits:
        bulk_lines = []

        for hit in hits:
            total_seen += 1

            doc_id = hit["_id"]
            source = hit.get("_source", {})

            old_file_url = source.get("file_url")
            new_file_url = normalize_file_url(old_file_url)

            if new_file_url != old_file_url:
                total_changed += 1

                if len(examples) < 10:
                    examples.append((old_file_url, new_file_url))

                if not args.dry_run:
                    update_doc = {
                        "file_url": new_file_url
                    }

                    # Preserva o caminho físico original no índice clonado.
                    # Isso não atrapalha a avaliação, porque o IQLoc usa file_url.
                    if "file_url_original" not in source:
                        update_doc["file_url_original"] = old_file_url

                    bulk_lines.append(json.dumps({
                        "update": {
                            "_index": args.index,
                            "_id": doc_id
                        }
                    }))

                    bulk_lines.append(json.dumps({
                        "doc": update_doc
                    }))

        if bulk_lines and not args.dry_run:
            ndjson_text = "\n".join(bulk_lines) + "\n"

            bulk_response = request_ndjson(
                "POST",
                f"{base_url}/_bulk",
                ndjson_text,
            )

            if bulk_response.get("errors"):
                print("Alguns documentos tiveram erro no bulk update.")
                print(json.dumps(bulk_response, indent=2, ensure_ascii=False))
                raise SystemExit(1)

            total_updated += len(bulk_lines) // 2

        if not scroll_id:
            break

        next_response = request_json(
            "POST",
            f"{base_url}/_search/scroll",
            {
                "scroll": "2m",
                "scroll_id": scroll_id
            },
        )

        scroll_id = next_response.get("_scroll_id")
        hits = next_response.get("hits", {}).get("hits", [])

    print("\nExemplos de conversão:")
    for old, new in examples:
        print("-" * 80)
        print("ANTES:", old)
        print("DEPOIS:", new)

    print("\nResumo:")
    print(f"Documentos lidos: {total_seen}")
    print(f"Documentos que precisam mudar: {total_changed}")

    if args.dry_run:
        print("Modo dry-run ativado: nada foi alterado no Elasticsearch.")
    else:
        print(f"Documentos atualizados: {total_updated}")


if __name__ == "__main__":
    main()
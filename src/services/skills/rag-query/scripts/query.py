
import sys
import json
import argparse
import httpx

# RAG 接口地址
RAG_ENDPOINT = "http://localhost:8000/api/rag/query"


def parse_args():
    parser = argparse.ArgumentParser(description="Query RAG knowledge base")
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask the RAG system"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top communities to retrieve"
    )
    return parser.parse_args()


def query_rag(question: str, top_k: int = 10) -> dict:
    payload = {
        "question": question,
        "top_k_communities": top_k
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            RAG_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload
        )

        response.raise_for_status()
        return response.json()


def main():
    args = parse_args()

    try:
        result = query_rag(args.question, args.top_k)

        print("=" * 60)
        print("📘 RAG Query Result")
        print("=" * 60)
        print(f"❓ Question:\n{args.question}\n")

        if result.get("success"):
            print("✅ Answer:\n")
            print(result.get("answer", ""))
            if "processing_time" in result:
                print(f"\n⏱ Processing Time: {result['processing_time']:.2f}s")
        else:
            print("❌ Query failed")
            print(json.dumps(result, ensure_ascii=False, indent=2))

    except httpx.HTTPError as e:
        print("❌ HTTP request failed")
        print(str(e))
        sys.exit(1)

    except Exception as e:
        print("❌ Unexpected error")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

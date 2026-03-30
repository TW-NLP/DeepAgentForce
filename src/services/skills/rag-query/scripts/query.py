import os
import sys
import json
import argparse
import httpx
# 添加config路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, ROOT)
from config import settings

# RAG 接口地址
RAG_ENDPOINT = settings.RAG_API_URL


def parse_args():
    parser = argparse.ArgumentParser(description="Query RAG knowledge base")

    # 位置参数（主要参数）
    parser.add_argument(
        "question_positional",
        type=str,
        nargs="?",
        help="Question to ask the RAG system (positional argument)"
    )

    # 兼容旧格式的参数
    parser.add_argument(
        "--query",
        type=str,
        help="Query parameter (alternative to positional question)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question parameter (alternative to positional question)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top communities to retrieve"
    )
    # 🆕 多租户参数
    parser.add_argument(
        "--tenant-uuid",
        type=str,
        default=None,
        help="Tenant UUID for multi-tenant RAG query"
    )

    args = parser.parse_args()

    # 优先级：positional > --question > --query
    if args.question_positional:
        final_question = args.question_positional
    elif args.question:
        final_question = args.question
    elif args.query:
        final_question = args.query
    else:
        final_question = None

    return argparse.Namespace(
        question=final_question,
        top_k=args.top_k,
        tenant_uuid=args.tenant_uuid
    )


def query_rag(question: str, top_k: int = 10, tenant_uuid: str = None) -> dict:
    payload = {
        "question": question,
        "top_k": top_k
    }
    # 🆕 添加租户标识（如果 RAG API 支持通过 header 传递）
    headers = {"Content-Type": "application/json"}
    if tenant_uuid:
        headers["X-Tenant-UUID"] = tenant_uuid

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            RAG_ENDPOINT,
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        return response.json()


def main():
    args = parse_args()

    # 检查问题是否提供
    if not args.question:
        print("❌ Error: No question provided.")
        print("Usage: python query.py \"Your question here\"")
        print("   or: python query.py --query \"Your question here\"")
        print("   or: python query.py --question \"Your question here\"")
        sys.exit(1)

    # 🆕 检查租户 UUID
    if not args.tenant_uuid:
        print("⚠️ Warning: No tenant_uuid provided, will query default RAG collection.")
        print("   Add --tenant-uuid <uuid> to query tenant-specific collection.")

    try:
        result = query_rag(args.question, args.top_k, args.tenant_uuid)

        print("=" * 60)
        print("📘 RAG Query Result")
        print(f"🏢 Tenant: {args.tenant_uuid or 'default'}")
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

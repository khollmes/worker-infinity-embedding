

import runpod
from utils import create_error_response
from typing import Any
from embedding_service import EmbeddingService

try:
    embedding_service = EmbeddingService()
except Exception as e:
    import sys
    sys.stderr.write(f"\nstartup failed: {e}\n")
    sys.exit(1)


def _to_jsonable(x):
    if isinstance(x, (dict, list, str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    if hasattr(x, "__dict__"):
        return x.__dict__
    return {"value": str(x)}


async def async_generator_handler(job: dict[str, Any]):
    job_input = job.get("input", {})
    print('test is in')
    if job_input.get("openai_route"):
        openai_route = job_input.get("openai_route")
        openai_input = job_input.get("openai_input")

        if not openai_input:
            return create_error_response("Missing openai_input").model_dump()

        if openai_route == "/v1/models":
            call_fn, kwargs = embedding_service.route_openai_models, {}

        elif openai_route == "/v1/embeddings":
            model_name = openai_input.get("model")
            if not model_name:
                return create_error_response("Did not specify model in openai_input").model_dump()

            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": openai_input.get("input"),
                "model_name": model_name,
                "return_as_list": True,
            }

        else:
            return create_error_response(f"Invalid OpenAI Route: {openai_route}").model_dump()

    else:
        if job_input.get("query"):
            print('rerank called')
            call_fn, kwargs = embedding_service.infinity_rerank, {
                "query": job_input.get("query"),
                "docs": job_input.get("docs"),
                "return_docs": job_input.get("return_docs", False),
                "model_name": job_input.get("model"),
            }
        elif job_input.get("input"):
            print('simple embedding')
            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": job_input.get("input"),
                "model_name": job_input.get("model"),
                "return_as_list": True,
            }
        else:
            return create_error_response(f"Invalid input: {job}").model_dump()

    try:
        out = await call_fn(**kwargs)
        print(out)
        return _to_jsonable(out)   
    except Exception as e:
        return create_error_response(str(e)).model_dump()


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": async_generator_handler,
            "concurrency_modifier": lambda x: embedding_service.config.runpod_max_concurrency,
        }
    )


import runpod
from runpod import RunPodLogger
log = RunPodLogger()
log.info("Early test for debugging")

from utils import create_error_response
from typing import Any, Optional, List, Union
from embedding_service import EmbeddingService
from pydantic import BaseModel, Field, ValidationError


log.info("handler module imported")

# Lazy init instead of creating at import time
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        log.info("initializing EmbeddingService...")
        try:
            _embedding_service = EmbeddingService()
            log.info("EmbeddingService initialized")
        except Exception as e:
            log.info(f"EmbeddingService startup failed: {e}")
            # re-raise so caller knows initialization failed
            raise
    return _embedding_service


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

class InputModel(BaseModel):
    # Use input_ as the attribute name and accept "input" from incoming dict
    input_: Optional[Union[str, List[Any], dict]] = Field(None, alias="input")
    openai_route: Optional[str] = None
    openai_input: Optional[dict] = None
    query: Optional[str] = None
    docs: Optional[List[Any]] = None
    model: Optional[str] = None
    return_docs: Optional[bool] = False


async def async_generator_handler(job: dict[str, Any]):
    log.info("async_generator_handler called")
    # Validate job input early with pydantic
    raw_input = job.get("input", {})
    try:
        job_input = InputModel.parse_obj(raw_input)
    except ValidationError as ve:
        err_msg = f"Invalid job input: {ve}"
        log.info(err_msg)
        raise ValueError(err_msg)
    except Exception as e:
        # Unexpected error during parsing
        log.info(f"Unexpected error parsing input: {e}")
        raise RuntimeError(str(e))

    # now get the embedding service (may raise, and will be logged by get_embedding_service)
    try:
        embedding_service = get_embedding_service()
    except Exception as e:
        raise RuntimeError(f"service init failed: {e}") from e

    # keep client informed that the async job started
    log.info("in async_generator_handler (after validation)")

    # handle "OpenAI route" style requests
    if job_input.openai_route:
        yield "openai_route"
        openai_route = job_input.openai_route
        openai_input = job_input.openai_input

        if not openai_input:
            raise ValueError("Missing openai_input")

        if openai_route == "/v1/models":
            call_fn, kwargs = embedding_service.route_openai_models, {}

        elif openai_route == "/v1/embeddings":
            model_name = openai_input.get("model")
            if not model_name:
                raise ValueError("Did not specify model in openai_input")

            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": openai_input.get("input"),
                "model_name": model_name,
                "return_as_list": True,
            }

        else:
            raise ValueError(f"Invalid OpenAI Route: {openai_route}")

    # handle other input types
    else:
        if job_input.query:
            log.info("in rerank")
            call_fn, kwargs = embedding_service.infinity_rerank, {
                "query": job_input.query,
                "docs": job_input.docs,
                "return_docs": job_input.return_docs or False,
                "model_name": job_input.model,
            }
        elif job_input.input_ is not None:
            log.info("in embedding")
            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": job_input.input_,
                "model_name": job_input.model,
                "return_as_list": True,
            }
        else:
            raise ValueError(f"Invalid input: {job}")

    # execute the chosen function and stream the result
    try:
        out = await call_fn(**kwargs)
        out_json = _to_jsonable(out)
        yield out_json
        return
    except Exception as e:
        log.info("handler error during execution")
        raise RuntimeError(str(e)) from e


if __name__ == "__main__":
    log.info("in main")
    # ensure embedding service is created when running standalone so concurrency lambda can access config
    try:
        es = get_embedding_service()
    except Exception:
        log.error("failed to initialize embedding service in __main__")
        raise

    runpod.serverless.start(
        {
            "handler": async_generator_handler,
            "concurrency_modifier": lambda x: es.config.runpod_max_concurrency,
            "return_aggregate_stream": True,
        }
    )

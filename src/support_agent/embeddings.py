"""Azure OpenAI embeddings client — shared between corpus generation
(`data/generate.py`) and query-time retrieval. Settings are only read when
`embed_texts` is actually called, so importing this module never requires
Azure credentials to be configured.
"""

from __future__ import annotations

from openai import AzureOpenAI

from support_agent.config import get_settings


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    response = client.embeddings.create(
        model=settings.azure_openai_deployment_embedding, input=texts
    )
    return [item.embedding for item in response.data]

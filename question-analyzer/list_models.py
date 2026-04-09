import asyncio
from openai import AsyncOpenAI
from app.core.config import settings

async def list_models():
    client = AsyncOpenAI(api_key=settings.GROK_API_KEY, base_url="https://api.x.ai/v1")
    models = await client.models.list()
    for m in models.data:
        print(m.id)

if __name__ == "__main__":
    asyncio.run(list_models())

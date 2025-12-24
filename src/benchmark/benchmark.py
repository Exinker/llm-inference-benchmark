import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any

import aiohttp
import pandas as pd
from openai import AsyncClient
from openai.types import Completion
from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmark.types import Second


class TestCase(BaseModel):

    profile: Sequence[int]
    messages: Sequence[Mapping[str, str]]
    temperature: float = Field(1)
    max_tokens: int = Field(128)


class Result(BaseModel):

    prompt_tokens: float
    completion_tokens: float
    elapsed_time: Second


def wakeup(method):

    async def inner(self: 'Benchmark', *args, **kwargs):

        compiltion = await self._complete(
            messages=[
                {'role': 'system', 'content': 'You are helpful assistant. Answer on the request.'},
                {'role': 'user', 'content': 'Write a "Hello World" script in Python 3.12.'},
            ],
            temperature=1.0,
            max_tokens=4096,
        )
        print(f'Wake up complition: {compiltion.choices[0].message.content}')

        return await method(self, *args, **kwargs)

    return inner


class Benchmark:

    def __init__(
        self,
        client: AsyncClient,
        model: str,
        n_requests: int = 10,
    ) -> None:

        self.client = client
        self.model = model
        self.n_requests = n_requests

    async def fetch_models(self) -> Mapping[str, Any]:

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url=str(self.client.base_url.join('models')),
                headers={
                    'Authorization': 'Bearer {api_key}'.format(
                        api_key=self.client.api_key,
                    ),
                },
            ) as response:
                response.raise_for_status()

                data = (await response.json())['data']

        return data

    @wakeup
    async def run(
        self,
        test_case: TestCase,
    ) -> pd.DataFrame:
        results = []

        started_at = time.perf_counter()
        for n_workers in tqdm(test_case.profile):
            results.append(await self._run_workers(
                test_case=test_case,
                n_workers=n_workers,
            ))
        print(f'Total time elapsed: {time.perf_counter() - started_at:.2f}, sec')

        report = pd.DataFrame(
            [
                result.model_dump()
                for result in results
            ],
            index=test_case.profile,
        )
        return report

    async def _run_workers(
        self,
        test_case: TestCase,
        n_workers: int,
    ) -> pd.Series:

        results = await asyncio.gather(*[
            self._run_worker(
                test_case=test_case,
            )
            for _ in range(n_workers)
        ])

        data = pd.DataFrame([
                result.model_dump()
                for result in results
            ],
        )
        result = Result(**(data.mean() / self.n_requests).to_dict())
        return result

    async def _run_worker(
        self,
        test_case: TestCase,
    ) -> pd.Series:
        completions = []

        started_at = time.perf_counter()
        for _ in range(self.n_requests):
            completion = await self._complete(
                messages=test_case.messages,
                temperature=test_case.temperature,
                max_tokens=test_case.max_tokens,
            )
            completions.append(completion)
        elapsed_time = time.perf_counter() - started_at

        data = pd.DataFrame([
                completion.usage.to_dict()
                for completion in completions
            ],
        )
        result = Result(
            prompt_tokens=data['prompt_tokens'].sum(),
            completion_tokens=data['completion_tokens'].sum(),
            elapsed_time=elapsed_time,
        )
        return result

    async def _complete(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Completion:

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion

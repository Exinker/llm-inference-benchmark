import asyncio
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from faker import Faker
from openai import AsyncClient

from benchmark import Benchmark, TestCase
from benchmark.configs import BENCHMARK_CONFIG, CLIENT_CONFIG

Faker.seed(BENCHMARK_CONFIG.seed)

REPORT_DIR = Path.cwd() / 'reports' / str(time.monotonic_ns())
REPORT_DIR.mkdir(parents=True, exist_ok=True)


async def main(
    benchmark: Benchmark,
    test_cases: Mapping[str, TestCase],
) -> None:

    models = await benchmark.fetch_models()
    filename = REPORT_DIR / '.info'
    with open(filename, 'w') as file:
        json.dump({
            'datetime': datetime.strftime(datetime.now(tz=timezone.utc), '%Y-%m-%d %H:%M:%S'),
            'model': CLIENT_CONFIG.model_name,
            'server': {
                'info': BENCHMARK_CONFIG.info,
                'host': CLIENT_CONFIG.host,
                'models': models,
            },
        }, file, indent=2)

    for test_name, test_case in test_cases.items():
        report = await benchmark.run(
            test_case=test_case,
        )

        filepath = REPORT_DIR / '{name}-stage.csv'.format(
            name=test_name,
        )
        report.to_csv(filepath)


def load_prompt(
    filename: str,
) -> str:
    faker = Faker()

    if filename:
        filepath = Path.cwd() / 'data' / filename

        with open(filepath, mode='r', encoding='utf-8') as file:
            prefil_prompt = file.read()
        return prefil_prompt

    prefil_prompt = ' '.join(faker.paragraphs(10))
    return prefil_prompt


if __name__ == '__main__':

    client = AsyncClient(
        base_url=CLIENT_CONFIG.endpoint_url,
        api_key=CLIENT_CONFIG.api_key.get_secret_value(),
    )
    benchmark = Benchmark(
        client=client,
        model=CLIENT_CONFIG.model_name,
    )

    profile = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    test_cases = {
        'prefill': TestCase(
            messages=[
                {'role': 'system', 'content': 'You are helpful assistant. Summarize a given text.'},
                {'role': 'user', 'content': load_prompt(
                    filename=BENCHMARK_CONFIG.prefill_prompt,
                )},
            ],
            temperature=0.2,
            max_tokens=1,
            profile=profile,
        ),
        'decode': TestCase(
            messages=[
                {'role': 'user', 'content': '?'},
            ],
            temperature=1,
            max_tokens=512,
            profile=profile,
        ),
    }
    asyncio.run(main(
        benchmark=benchmark,
        test_cases=test_cases,
    ))

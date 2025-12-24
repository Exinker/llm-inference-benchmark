import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path

from faker import Faker
from openai import AsyncClient

from benchmark import Benchmark, TestCase
from benchmark.configs import CLIENT_CONFIG


REPORT_DIR = Path.cwd() / 'reports' / str(time.monotonic_ns())
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FAKER = Faker()


async def main(
    benchmark: Benchmark,
    test_cases: Mapping[str, TestCase],
) -> None:

    with open(REPORT_DIR / '.info', 'w') as file:
        json.dump({
            'model': CLIENT_CONFIG.model_name,
            'server': {
                'host': CLIENT_CONFIG.host,
                'models': await benchmark.fetch_models(),
                'info': CLIENT_CONFIG.info,
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
        'prefilling': TestCase(
            messages=[
                {'role': 'system', 'content': 'You are helpful assistant. Summarize a given text.'},
                {'role': 'user', 'content': FAKER.paragraph(10)},
            ],
            temperature=0.2,
            max_tokens=1,
            profile=profile,
        ),
        'decoding': TestCase(
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

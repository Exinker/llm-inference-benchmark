from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkConfig(BaseSettings):

    prefill_prompt: str | None = Field(None, alias='BENCHMARK_PREFILL_PROMPT')
    seed: int = Field(42, alias='BENCHMARK_SEED')
    info: str = Field('', alias='BENCHMARK_INFO')

    model_config = SettingsConfigDict(
        env_file = '.env',
        env_file_encoding='utf-8',
        extra = 'ignore',
    )

    @computed_field
    @property
    def endpoint_url(self) -> str:
        return 'http://{host}:{port}/v1'.format(
            host=self.host,
            port=self.port,
        )


BENCHMARK_CONFIG = BenchmarkConfig()

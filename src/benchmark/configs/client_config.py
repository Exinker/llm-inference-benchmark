from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClientConfig(BaseSettings):

    host: str = Field(alias='CLIENT_HOST')
    port: int = Field(alias='CLIENT_PORT')
    api_key: SecretStr = Field(alias='CLIENT_API_KEY')
    model_name: str = Field(alias='CLIENT_MODEL_NAME')
    info: str = Field('', alias='CLIENT_INFO')

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


CLIENT_CONFIG = ClientConfig()

import ast
import os
import sys

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide


class ConfigService():
    config: dict = {}

    def config(self, config: dict):
        self.config = config

    def property(self, key: str, default: any = None) -> any:
        value = os.getenv(key, self.__get_from_dict(key))
        if value is None:
            value = default

        if isinstance(default, bool) and not isinstance(value, bool):
            value = ast.literal_eval(value)

        return value

    def __get_from_dict(self, path: str) -> str:
        value: str = self.config
        for key in path.split('.'):
            if value:
                value = value.get(key)
            else:
                return None

        return value


class ConfigContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    config_svc = providers.Singleton(
        ConfigService,
    )


def initialize(config_file: str, module: str) -> ConfigContainer:
    container: ConfigContainer = ConfigContainer()
    container.config.from_yaml(config_file)
    container.wire(modules=[sys.modules[module]])

    config_provider = Provide[ConfigContainer.config_svc]
    config_provider.provider().config(container.config.provided())

    return container

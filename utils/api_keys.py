import os
from dataclasses import dataclass
from typing import Iterable


def load_groq_api_keys(prefix: str = "GROQ_API_KEY", slots: int = 6) -> list[str]:
    keys: list[str] = []
    for index in range(1, slots + 1):
        value = os.getenv(f"{prefix}_{index}")
        if value:
            keys.append(value)

    if not keys:
        direct_key = os.getenv(prefix)
        if direct_key:
            keys.append(direct_key)

    return keys


@dataclass
class ApiKeyManager:
    keys: list[str]
    index: int = 0

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("At least one API key is required")

    @classmethod
    def from_value(cls, api_key: str | Iterable[str]) -> "ApiKeyManager":
        if isinstance(api_key, str):
            return cls(keys=[api_key])
        return cls(keys=[k for k in api_key if k])

    @property
    def current(self) -> str:
        return self.keys[self.index]

    @property
    def position(self) -> int:
        return self.index + 1

    def rotate(self) -> bool:
        self.index += 1
        return self.index < len(self.keys)

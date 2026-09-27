"""Shared HTTP helper so every tool handles errors the same way."""

import requests

from finagent import config


def get_json(url: str, params: dict | None = None,
             headers: dict | None = None) -> dict | list:
    resp = requests.get(
        url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def get_text(url: str, headers: dict | None = None) -> str:
    resp = requests.get(url, headers=headers, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.text

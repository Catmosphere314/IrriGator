"""Authentication data vault."""

from __future__ import annotations

__all__ = [
    "Credential",
    "clear",
    "clear_meteofrance_token",
    "configure_meteofrance_client_auth",
    "export",
    "get",
    "get_meteofrance_access_token",
    "load",
    "meteo_headers",
    "override",
]

import os
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from enum import Enum
from getpass import getpass

import requests

from .crypt import decrypt, encrypt
from .io import cache_create

_cache = cache_create(__name__)

# Météo-France OAuth token endpoint.
# Keep configurable so tests or future API changes do not require code changes.
METEOFRANCE_TOKEN_URL = os.environ.get(
    "METEOFRANCE_TOKEN_URL",
    "https://portail-api.meteofrance.fr/token",
)

# In-memory only. Do not persist short-lived OAuth tokens.
_METEOFRANCE_TOKEN: str | None = None
_METEOFRANCE_TOKEN_EXPIRES_AT = 0.0


class Credential(str, Enum):
    """Project credentials."""
    ECMWF_DATASTORE_KEY = "ecmwf_datastore_key"
    METEOFRANCE_CLIENT_AUTH = "meteofrance_client_auth"  # noqa: S105


_SECRET_CREDENTIALS = {
    Credential.ECMWF_DATASTORE_KEY,
    Credential.METEOFRANCE_CLIENT_AUTH,
}


def _input(prompt: str, *, secret: bool = False) -> str | None:
    """Input value or secret."""
    if secret:
        return getpass(prompt)
    return input(prompt)


def _coerce_credential(credential: Credential | str) -> Credential:
    """Convert enum, enum value, or enum name to a Credential."""
    if isinstance(credential, Credential):
        return credential

    try:
        return Credential(credential)
    except ValueError:
        return Credential[credential]


def _clean_meteofrance_client_auth(value: str) -> str:
    """Accept either '<client-auth>' or 'Basic <client-auth>'."""
    value = value.strip()
    if value.lower().startswith("basic "):
        value = value.split(None, 1)[1].strip()
    return value


def override(credential: Credential | str, value: str) -> None:
    """Set credential in cache.

    Args:
        credential: The credential to set.
        value: The value to set.

    """
    credential = _coerce_credential(credential)

    if credential is Credential.METEOFRANCE_CLIENT_AUTH:
        value = _clean_meteofrance_client_auth(value)

    _cache.set(credential.value, value)


def get(credential: Credential | str, *, allow_empty: bool = False) -> str:
    """Retrieve credential from env or cache.

    Lookup order:
    1. Environment variable using the enum name, e.g. METEOFRANCE_CLIENT_AUTH.
    2. Local credential cache.
    3. Secure prompt for secret credentials.

    Args:
        credential: The credential to retrieve.
        allow_empty: Whether to allow empty values.

    """
    credential = _coerce_credential(credential)
    is_secret = credential in _SECRET_CREDENTIALS

    value = os.environ.get(credential.name)
    if value is not None:
        if credential is Credential.METEOFRANCE_CLIENT_AUTH:
            value = _clean_meteofrance_client_auth(value)
        return value

    value = _cache.get(credential.value)
    if value is not None:
        if credential is Credential.METEOFRANCE_CLIENT_AUTH:
            value = _clean_meteofrance_client_auth(value)
        return value

    value = _input(f"Enter {credential.name}: ", secret=is_secret)
    if not allow_empty and not value:
        msg = "Empty credentials are not allowed"
        raise ValueError(msg)
    if value is None:
        return ""

    if credential is Credential.METEOFRANCE_CLIENT_AUTH:
        value = _clean_meteofrance_client_auth(value)

    override(credential, value)
    return value


def configure_meteofrance_client_auth(*, force_prompt: bool = False) -> str:
    """Ensure METEOFRANCE_CLIENT_AUTH exists in the current process.

    This stores the long-lived Météo-France client auth secret in the same local
    cache used by the other project credentials. It also injects it into
    os.environ for libraries that expect METEOFRANCE_CLIENT_AUTH.

    Args:
        force_prompt: Prompt again and overwrite the cached value.

    Returns:
        The cleaned Météo-France client auth value, without the leading
        'Basic ' prefix.

    """
    if force_prompt:
        value = _input("Enter METEOFRANCE_CLIENT_AUTH: ", secret=True)
        if not value:
            msg = "Empty credentials are not allowed"
            raise ValueError(msg)
        value = _clean_meteofrance_client_auth(value)
        override(Credential.METEOFRANCE_CLIENT_AUTH, value)
    else:
        value = get(Credential.METEOFRANCE_CLIENT_AUTH)

    os.environ[Credential.METEOFRANCE_CLIENT_AUTH.name] = value
    return value


def get_meteofrance_access_token(*, force_refresh: bool = False) -> str:
    """Return a valid Météo-France OAuth bearer token.

    The access token is cached in memory only and refreshed automatically before
    expiry. Only METEOFRANCE_CLIENT_AUTH is stored in the credential cache.

    Args:
        force_refresh: Ignore the in-memory token cache and request a new token.

    Returns:
        A valid OAuth access token.

    """
    global _METEOFRANCE_TOKEN, _METEOFRANCE_TOKEN_EXPIRES_AT  # noqa: PLW0603

    now = time.time()
    if (
        not force_refresh
        and _METEOFRANCE_TOKEN
        and now < _METEOFRANCE_TOKEN_EXPIRES_AT - 60
    ):
        return _METEOFRANCE_TOKEN

    client_auth = configure_meteofrance_client_auth()

    response = requests.post(
        METEOFRANCE_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {client_auth}"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    access_token = payload["access_token"]
    expires_in = int(payload.get("expires_in", 3600))

    _METEOFRANCE_TOKEN = access_token
    _METEOFRANCE_TOKEN_EXPIRES_AT = now + expires_in

    return access_token


def meteo_headers() -> dict[str, str]:
    """Return authorization headers for Météo-France API calls."""
    return {"Authorization": f"Bearer {get_meteofrance_access_token()}"}


def clear_meteofrance_token() -> None:
    """Clear the in-memory Météo-France bearer token only."""
    global _METEOFRANCE_TOKEN, _METEOFRANCE_TOKEN_EXPIRES_AT  # noqa: PLW0603
    _METEOFRANCE_TOKEN = None
    _METEOFRANCE_TOKEN_EXPIRES_AT = 0.0


def clear() -> None:
    """Clear cached credentials and in-memory tokens."""
    _cache.clear()
    clear_meteofrance_token()


def export(password: str) -> dict[str, str]:
    """Export encrypted credentials.

    Args:
        password: The password to use for encryption.

    Returns:
        A dictionary of encrypted credentials.

    """
    return {
        credential.value: urlsafe_b64encode(
            encrypt(get(credential).encode(), password)
        ).decode()
        for credential in Credential
    }


def load(
    data: dict[str, str],
    password: str | None = None,
    *,
    force: bool = False,
) -> None:
    """Load encrypted credentials.

    Args:
        data: A dictionary of encrypted credentials.
        password: The password to use for decryption.
        force: Whether to override existing credentials.

    """
    if password is None:
        password = _input("Enter password to decrypt credentials: ", secret=True)
    for credential in map(Credential, data):
        encrypted = urlsafe_b64decode(data[credential.value].encode())
        value = decrypt(encrypted, password or "").decode()
        if not force and credential.value in _cache:
            if _cache[credential.value] == value:
                continue
            msg = (
                f"Credential {credential.value!r} is already set."
                " Use force=True to override."
            )
            raise RuntimeError(msg)
        override(credential, value)

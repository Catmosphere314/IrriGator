import base64
import getpass
import os
import time
from typing import Optional

import keyring
import requests


ENV_NAME = "METEOFRANCE_CLIENT_AUTH"
KEYRING_SERVICE = "meteofrance-api"
KEYRING_USERNAME = "client_auth"

TOKEN_URL = "https://portail-api.meteofrance.fr/token"

_cached_token: Optional[str] = None
_token_expiry: float = 0


def _build_basic_auth_from_id_pwd(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return base64.b64encode(raw).decode("utf-8")


def ensure_meteofrance_client_auth() -> str:
    """
    Returns METEOFRANCE_CLIENT_AUTH securely.

    Priority:
    1. Existing environment variable
    2. User OS keychain
    3. Interactive prompt, then save to OS keychain

    The secret is never written to the repo or to a plaintext file.
    """

    existing = os.getenv(ENV_NAME)
    if existing:
        return existing

    stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if stored:
        os.environ[ENV_NAME] = stored
        return stored

    print("Météo-France credentials not found.")
    print("Choose one option:")
    print("1. Paste the value after 'Authorization: Basic ...'")
    print("2. Enter client ID and client secret/password")

    choice = input("Option [1/2]: ").strip()

    if choice == "1":
        client_auth = getpass.getpass("METEOFRANCE_CLIENT_AUTH: ").strip()

    elif choice == "2":
        client_id = input("Météo-France client ID: ").strip()
        client_secret = getpass.getpass("Météo-France client secret/password: ").strip()
        client_auth = _build_basic_auth_from_id_pwd(client_id, client_secret)

    else:
        raise ValueError("Invalid option. Choose 1 or 2.")

    if not client_auth:
        raise ValueError("Empty Météo-France credential.")

    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, client_auth)
    os.environ[ENV_NAME] = client_auth

    return client_auth


def get_meteofrance_token() -> str:
    """
    Returns a valid Météo-France access token.
    Automatically refreshes the token when needed.
    """

    global _cached_token, _token_expiry

    if _cached_token and time.time() < _token_expiry - 60:
        return _cached_token

    client_auth = ensure_meteofrance_client_auth()

    response = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {client_auth}"},
        timeout=20,
    )
    response.raise_for_status()

    payload = response.json()

    _cached_token = payload["access_token"]
    _token_expiry = time.time() + int(payload.get("expires_in", 3600))

    return _cached_token


def meteofrance_get(url: str, **params) -> requests.Response:
    """
    Authenticated GET request to a Météo-France API endpoint.
    Retries once if the token expired.
    """

    global _cached_token, _token_expiry

    token = get_meteofrance_token()

    response = requests.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )

    if response.status_code == 401:
        _cached_token = None
        _token_expiry = 0

        token = get_meteofrance_token()

        response = requests.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )

    response.raise_for_status()
    return response

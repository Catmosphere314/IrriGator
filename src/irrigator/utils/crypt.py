"""Utilities for encryption and decryption of data."""

import os
from base64 import urlsafe_b64encode
from hashlib import pbkdf2_hmac

from cryptography.fernet import Fernet


def _fernet(password: str, salt: bytes) -> Fernet:
    key = pbkdf2_hmac("sha256", password.encode(), salt, 100000, dklen=32)
    return Fernet(urlsafe_b64encode(key))


def encrypt(data: bytes, password: str) -> bytes:
    """Encrypt the given data using the provided password.

    Args:
        data: The data to encrypt.
        password: The password to use for encryption.

    Returns:
        The encrypted data.

    """
    salt = os.urandom(16)
    fernet = _fernet(password, salt)
    return salt + fernet.encrypt(data)


def decrypt(data: bytes, password: str) -> bytes:
    """Decrypt the given data using the provided password.

    Args:
        data: The data to decrypt.
        password: The password to use for decryption.

    Returns:
        The decrypted data.

    """
    salt, data = data[:16], data[16:]
    fernet = _fernet(password, salt)
    return fernet.decrypt(data)

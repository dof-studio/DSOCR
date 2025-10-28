# Project DSOCR
#
# dsocr_crypto.py
# Providing Decryptive Cryptography utils for DS OCR
# by dof-studio/Nathmath
# Open Source Under Apache 2.0 License
# Website: https://github.com/dof-studio/DSOCR


from Crypto.Util.Padding import unpad
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES
from hashlib import blake2b

def int_to_bytes(i: int, length: int) -> bytes:
    return i.to_bytes(length, byteorder='big')

def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, byteorder='big')

def blake2b_derive_key(data: bytes, key_bytes: int = 32) -> bytes:
    h = blake2b(digest_size=key_bytes)
    h.update(data)
    return h.digest()

def import_rsa_key(pem):
    return RSA.import_key(pem)

def rsa_public_decrypt_raw(rsa_pubkey: RSA.RsaKey, ciphertext: bytes) -> bytes:
    """
    Perform raw RSA public-key operation: m = c^e mod n
    """
    n = rsa_pubkey.n
    e = rsa_pubkey.e
    k = (n.bit_length() + 7) // 8
    c_int = bytes_to_int(ciphertext)
    m_int = pow(c_int, e, n)
    # strip leading zero bytes that may appear due to fixed length k
    m_bytes = int_to_bytes(m_int, k).lstrip(b'\x00')
    return m_bytes

def aes_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = cipher.decrypt(ciphertext)
    plaintext = unpad(padded, AES.block_size)
    return plaintext[16:]

import ctypes
import sys


class CredentialStoreError(RuntimeError):
    """Raised when a credential backend cannot complete an operation."""


class CredentialStoreUnavailable(CredentialStoreError):
    """Raised when no usable system credential backend is available."""


class UnavailableCredentialStore:
    def __init__(self, reason="No system credential backend is available"):
        self.reason = reason

    def set_secret(self, credential_ref, secret):
        raise CredentialStoreUnavailable(self.reason)

    def get_secret(self, credential_ref):
        raise CredentialStoreUnavailable(self.reason)

    def delete_secret(self, credential_ref):
        raise CredentialStoreUnavailable(self.reason)


class WindowsCredentialStore:
    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    ERROR_NOT_FOUND = 1168

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.c_ulong),
            ("dwHighDateTime", ctypes.c_ulong),
        ]

    class CREDENTIALW(ctypes.Structure):
        pass

    PCREDENTIALW = ctypes.POINTER(CREDENTIALW)

    CREDENTIALW._fields_ = [
        ("Flags", ctypes.c_ulong),
        ("Type", ctypes.c_ulong),
        ("TargetName", ctypes.c_wchar_p),
        ("Comment", ctypes.c_wchar_p),
        ("LastWritten", FILETIME),
        ("CredentialBlobSize", ctypes.c_ulong),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", ctypes.c_ulong),
        ("AttributeCount", ctypes.c_ulong),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", ctypes.c_wchar_p),
        ("UserName", ctypes.c_wchar_p),
    ]

    def __init__(self, service_name):
        if sys.platform != "win32":
            raise CredentialStoreUnavailable("Windows Credential Manager is only available on Windows")
        self.service_name = service_name
        self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._advapi32.CredWriteW.argtypes = [ctypes.POINTER(self.CREDENTIALW), ctypes.c_ulong]
        self._advapi32.CredWriteW.restype = ctypes.c_bool
        self._advapi32.CredReadW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.POINTER(self.PCREDENTIALW),
        ]
        self._advapi32.CredReadW.restype = ctypes.c_bool
        self._advapi32.CredDeleteW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong]
        self._advapi32.CredDeleteW.restype = ctypes.c_bool
        self._advapi32.CredFree.argtypes = [ctypes.c_void_p]
        self._advapi32.CredFree.restype = None

    def _target_name(self, credential_ref):
        return f"{self.service_name}:{credential_ref}"

    def set_secret(self, credential_ref, secret):
        target_name = self._target_name(credential_ref)
        secret_bytes = str(secret).encode("utf-16-le")
        blob = ctypes.create_string_buffer(secret_bytes)
        credential = self.CREDENTIALW()
        credential.Type = self.CRED_TYPE_GENERIC
        credential.TargetName = target_name
        credential.CredentialBlobSize = len(secret_bytes)
        credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
        credential.Persist = self.CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = credential_ref
        if not self._advapi32.CredWriteW(ctypes.byref(credential), 0):
            error = ctypes.get_last_error()
            raise CredentialStoreUnavailable(f"Windows Credential Manager write failed: {error}")

    def get_secret(self, credential_ref):
        target_name = self._target_name(credential_ref)
        credential_pointer = self.PCREDENTIALW()
        if not self._advapi32.CredReadW(
            target_name,
            self.CRED_TYPE_GENERIC,
            0,
            ctypes.byref(credential_pointer),
        ):
            error = ctypes.get_last_error()
            if error == self.ERROR_NOT_FOUND:
                return None
            raise CredentialStoreUnavailable(f"Windows Credential Manager read failed: {error}")
        try:
            credential = credential_pointer.contents
            blob = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return blob.decode("utf-16-le")
        finally:
            self._advapi32.CredFree(credential_pointer)

    def delete_secret(self, credential_ref):
        target_name = self._target_name(credential_ref)
        if not self._advapi32.CredDeleteW(target_name, self.CRED_TYPE_GENERIC, 0):
            error = ctypes.get_last_error()
            if error == self.ERROR_NOT_FOUND:
                return
            raise CredentialStoreUnavailable(f"Windows Credential Manager delete failed: {error}")


class KeyringCredentialStore:
    def __init__(self, service_name):
        self.service_name = service_name
        try:
            import keyring
        except Exception as e:
            raise CredentialStoreUnavailable(f"Python keyring unavailable: {type(e).__name__}") from e
        self._keyring = keyring

    def set_secret(self, credential_ref, secret):
        try:
            self._keyring.set_password(self.service_name, credential_ref, str(secret))
        except Exception as e:
            raise CredentialStoreUnavailable(f"Python keyring write failed: {type(e).__name__}") from e

    def get_secret(self, credential_ref):
        try:
            return self._keyring.get_password(self.service_name, credential_ref)
        except Exception as e:
            raise CredentialStoreUnavailable(f"Python keyring read failed: {type(e).__name__}") from e

    def delete_secret(self, credential_ref):
        try:
            self._keyring.delete_password(self.service_name, credential_ref)
        except Exception as e:
            error_name = type(e).__name__
            if error_name in {"PasswordDeleteError", "KeyringError"}:
                return
            raise CredentialStoreUnavailable(f"Python keyring delete failed: {error_name}") from e


def create_default_credential_store(service_name):
    if sys.platform == "win32":
        try:
            return WindowsCredentialStore(service_name)
        except CredentialStoreUnavailable as e:
            return UnavailableCredentialStore(str(e))

    try:
        return KeyringCredentialStore(service_name)
    except CredentialStoreUnavailable as e:
        return UnavailableCredentialStore(str(e))

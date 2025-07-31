# check_keyring.py
import keyring
import keyring.errors

def check_creds(service, username):
    """Checks for a password for a given service and username."""
    try:
        password = keyring.get_password(service, username)
        if password:
            print(f"SUCCESS: Found stored token for service='{service}', username='{username}'")
            return True
    except keyring.errors.NoKeyringError:
        print("CRITICAL: No system keyring backend found. This is not the source of the issue.")
        return True # Stop checking
    except Exception as e:
        print(f"WARNING: An error occurred for service='{service}': {e}")
    return False

print("Checking for stored credentials that twine might be using...")

services_to_check = ["testpypi", "https://test.pypi.org/legacy/", "test.pypi.org"]
usernames_to_check = ["__token__"]

found = False
for service in services_to_check:
    for username in usernames_to_check:
        if check_creds(service, username):
            found = True
            break
    if found:
        break

if not found:
    print("INFO: No commonly-named credentials found in the system keyring.")

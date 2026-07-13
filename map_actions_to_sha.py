import subprocess
import re
import glob

def get_latest_sha(repo_ref):
    repo, ref = repo_ref.split('@')
    try:
        if repo.startswith('r-lib/actions/'):
            repo_url = 'https://github.com/r-lib/actions.git'
        else:
            repo_url = f'https://github.com/{repo}.git'

        output = subprocess.check_output(['git', 'ls-remote', repo_url, ref], text=True)
        return output.split()[0]
    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None

actions = set()
for file in glob.glob('.github/workflows/*.yml'):
    with open(file, 'r') as f:
        content = f.read()
        for match in re.finditer(r'uses:\s*([^\s]+@[^\s]+)', content):
            actions.add(match.group(1))

mapping = {}
for action in actions:
    sha = get_latest_sha(action)
    if sha:
        mapping[action] = sha
        print(f"Mapped {action} to {sha}")
    else:
        print(f"Failed to map {action}")

with open('action_shas.txt', 'w') as f:
    for action, sha in mapping.items():
        f.write(f"{action} {sha}\n")

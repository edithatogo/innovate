import re
import glob

# Load mapping
mapping = {}
with open('action_shas.txt', 'r') as f:
    for line in f:
        action, sha = line.strip().split()
        mapping[action] = sha

def replace_action(match):
    action = match.group(1)
    if action in mapping:
        return f"uses: {action.split('@')[0]}@{mapping[action]}"
    return match.group(0)

for file in glob.glob('.github/workflows/*.yml'):
    with open(file, 'r') as f:
        content = f.read()

    # Replace uses: action@ref with uses: action@sha
    content = re.sub(r'uses:\s*([^\s]+@[^\s]+)', replace_action, content)

    lines = content.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        if re.search(r'uses:\s*actions/checkout@[0-9a-f]{40}', line):
            # Check if next line is `with:`
            has_with = False
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if re.match(r'^\s*with:\s*$', next_line):
                    has_with = True

            if has_with:
                new_lines.append(lines[i+1])
                # determine proper indentation for with block
                with_indent_match = re.match(r'^(\s*)', lines[i+1])
                with_indent = with_indent_match.group(1)
                new_lines.append(f"{with_indent}  persist-credentials: false")
                i += 1 # skip original `with:`
            else:
                uses_col = line.find('uses:')
                real_indent = " " * uses_col
                new_lines.append(f"{real_indent}with:")
                new_lines.append(f"{real_indent}  persist-credentials: false")

        i += 1

    with open(file, 'w') as f:
        f.write('\n'.join(new_lines))

print("Modified files.")

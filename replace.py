import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace st.session_state.page = 'X' with navigate_to('X')
content = re.sub(r"st\.session_state\.page\s*=\s*['\"]([^'\"]+)['\"]", r"navigate_to('\1')", content)

# Remove st.rerun() lines that might follow navigate_to
content = re.sub(r"(navigate_to\('[^']+'\))\s*\n\s*st\.rerun\(\)", r"\1", content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

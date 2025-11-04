from typing import List, Dict
import re


def split_markdown(text: str, max_chars: int = 1200, overlap: int = 120, min_chars: int = 200) -> List[Dict]:
# split by headings then slide
parts = re.split(r"\n(?=#+ )", text)
chunks = []
for part in parts:
start = 0
while start < len(part):
end = min(len(part), start + max_chars)
chunk = part[start:end]
if len(chunk) >= min_chars:
chunks.append({"text": chunk})
start = max(start + max_chars - overlap, end)
return chunks
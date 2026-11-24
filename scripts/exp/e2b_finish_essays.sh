#!/bin/bash
# Step 2b: RULER's essay downloader (urllib, sequential) fetched the 168 paulgraham.com pages fine but the 49 GitHub raw
# files at ~1/minute. Stopped it; fetch the missing GitHub files with curl (same URLs, raw bytes = same as the script's
# utf-8 decode), then assemble PaulGrahamEssays.json exactly as the script's lines 51-62 do.
L=~/kv-offload/logs/exp/e2b_finish_essays.log
{
cd ~/kv-offload/third_party/RULER/scripts/data/synthetic/json
pkill -f download_paulgraham && echo "stopped RULER downloader"; sleep 2
echo "before: html $(ls essay_html | wc -l), repo $(ls essay_repo | wc -l)"
grep -v '\.html' PaulGrahamEssays_URLs.txt | while read u; do
  f=essay_repo/$(basename "$u")
  [ -s "$f" ] || curl -sSfL --retry 3 --max-time 60 -o "$f" "$u" || echo "FAILED $u"
done
echo "after : html $(ls essay_html | wc -l) of $(grep -c '\.html' PaulGrahamEssays_URLs.txt), repo $(ls essay_repo | wc -l) of $(grep -vc '\.html' PaulGrahamEssays_URLs.txt)"
echo "html URLs without a file (failed in RULER's downloader):"
grep '\.html' PaulGrahamEssays_URLs.txt | while read u; do b=$(basename "$u" .html).txt; [ -e essay_html/$b ] || echo "  $u"; done
cd ~/kv-offload && . .venv/bin/activate && cd - >/dev/null
python - <<'EOF'
import glob, json, os
files_repo = sorted(glob.glob(os.path.join('essay_repo', '*.txt')))
files_html = sorted(glob.glob(os.path.join('essay_html', '*.txt')))
print(f'{len(files_repo)} essays from GitHub, {len(files_html)} from paulgraham.com')
text = ""
for file in files_repo + files_html:
    with open(file, 'r') as f:
        text += f.read()
with open('PaulGrahamEssays.json', 'w') as f:
    json.dump({"text": text}, f)
print('chars', len(text), 'words', len(text.split()))
EOF
ls -la PaulGrahamEssays.json; sha256sum PaulGrahamEssays.json
} > $L 2>&1
cat $L

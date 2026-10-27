import sys
import lancedb
import config

src, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
table = lancedb.connect(config.LANCEDB_URI).open_table(config.TABLE_NAME)
rows = table.search().select(["text", "metadata"]).limit(10_000_000).to_list()
for r in sorted(rows, key=lambda r: (r["metadata"]["source"], int(r["metadata"]["chunk_index"]))):
    m = r["metadata"]
    pages = [int(p) for p in str(m["pages"]).replace(" ", "").split(",") if p.isdigit()]
    if src.lower() in m["source"].lower() and any(first <= p <= last for p in pages):
        print(f"\n=== {m['source']}#{m['chunk_index']} p.{m['pages']} | {m['headings']}\n{r['text']}")
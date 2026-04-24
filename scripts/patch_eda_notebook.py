"""Patch eda.ipynb cell 6 — proper holiday lift analysis with locale-priority deduplication."""
import json
from pathlib import Path

nb_path = Path("notebooks/eda.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Root cause: holidays.csv has up to 3 rows per date (National + Regional + Local).
# set_index('date') creates a non-unique index -> map() raises InvalidIndexError.
# Fix: rank by locale priority (National=0 > Regional=1 > Local=2), keep highest-priority row per date.
new_source = [
    "# Cell 6: Holiday Lift Analysis\n",
    "# Root cause: holidays has multiple rows per date (National + Regional + Local layers).\n",
    "# Fix: take highest-priority locale per date (National > Regional > Local).\n",
    "locale_priority = {'National': 0, 'Regional': 1, 'Local': 2}\n",
    "h_deduped = (\n",
    "    holidays[holidays['transferred'] == False]\n",
    "    .copy()\n",
    "    .assign(priority=lambda x: x['locale'].map(locale_priority).fillna(99))\n",
    "    .sort_values(['date', 'priority'])\n",
    "    .drop_duplicates(subset='date', keep='first')\n",
    "    .set_index('date')[['type']]\n",
    ")\n",
    "\n",
    "df['holiday_type'] = df['date'].map(h_deduped['type']).fillna('non_holiday')\n",
    "\n",
    "holiday_avg = df.groupby('holiday_type')['sales'].mean().reset_index()\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.barplot(x='holiday_type', y='sales', data=holiday_avg)\n",
    "plt.title('Holiday Lift Analysis - Mean Sales by Holiday Type')\n",
    "plt.xticks(rotation=30, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.savefig(eda_dir / 'holiday_lift.png', dpi=150, bbox_inches='tight')\n",
    "plt.close()\n",
    "print('Saved: holiday_lift.png')\n",
    "print('Holiday types found:', df['holiday_type'].unique().tolist())",
]

patched = 0
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "holiday" in src.lower() and ("set_index" in src or "groupby" in src or "holiday_map" in src):
        cell["source"] = new_source
        patched += 1

print(f"Patched {patched} cell(s)")
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Written OK")

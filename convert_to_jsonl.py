import csv
import json

input_csv = "CarRacing-Data/actions.csv"
output_jsonl = "CarRacing-Data/caption_data.jsonl"

with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    data = [{"image": f"CarRacing-Data/images/{row['filename']}", "text": row["action_text"]}
            for row in reader]

with open(output_jsonl, "w", encoding="utf-8") as f:
    for entry in data:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ 変換完了！{len(data)} 件を {output_jsonl} に保存しました。")

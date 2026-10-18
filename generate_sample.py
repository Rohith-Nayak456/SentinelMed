import json
from pathlib import Path

from scan_queue.generator import QueueGenerator

project_dir = Path(__file__).resolve().parent
replay_file = project_dir / "scan_queue" / "replay_data.json"
output_file = project_dir / "sample_p5_queue.json"

queue_generator = QueueGenerator()
rows = queue_generator.generate_replay(str(replay_file))

with output_file.open("w", encoding="utf-8") as file:
  json.dump(rows, file, indent=2, ensure_ascii=False)

print(f"Queue records generated: {len(rows)}")
print(f"Output saved to: {output_file}")
print(f"Errors: {queue_generator.errors}")
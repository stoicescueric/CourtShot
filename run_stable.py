"""
Run Padex with stable player IDs (P_1..P_4) and gap filling.

From the padex folder:
    rm -f match_tracking_cache.pkl      # only when tracker settings or calibration changed
    uv run python run_stable.py

Output video: output/stable_ids.mp4
"""

import logging

from stable_ids import StableIdConfig, StablePadex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

VIDEO = "match.mp4"  # match_calibration.json next to it is found automatically

p = StablePadex(VIDEO, id_config=StableIdConfig(max_gap_fill=1.5))
result = p.run()
p.export_video(result, "output/stable_ids.mp4")

r = p.id_report
print("\n=== Player IDs ===")
print(f"frames with players:   {r.get('frames')}")
print(f"interpolated frames:   {r.get('interpolated_frames')}  (gaps filled, up to 1.5 s each)")
print(f"extra people dropped:  {r.get('dropped_detections')}")
for name, ids in r.get("tracker_ids_merged", {}).items():
    print(f"{name}: {len(ids)} raw tracker IDs merged -> {ids}")

print(f"\n=== Shots: {len(result.shots)} ===")
for s in result.shots:
    print(f"{s.timestamp_ms / 1000:7.2f}s  {s.player_id} ({s.team_id})  {s.shot_type.value}")
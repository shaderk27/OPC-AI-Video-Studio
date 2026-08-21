import os
import json
from moviepy.editor import VideoFileClip

OUTPUT_DIR = "output/"
CLIPS_DIR = os.path.join(OUTPUT_DIR, "formatted")
SUBS_DIR = os.path.join(OUTPUT_DIR, "subtitles")
REPORT_FILE = os.path.join(OUTPUT_DIR, "quality_report.json")

def evaluate_pipeline():
    print("--- 📊 Starting Pipeline Automated QA Check ---\n")
    
    # 1. Output Length Check (Must be 15~60 seconds)
    clips = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4")] if os.path.exists(CLIPS_DIR) else []
    length_passed = True
    for clip_name in clips:
        clip_path = os.path.join(CLIPS_DIR, clip_name)
        with VideoFileClip(clip_path) as video:
            duration = video.duration
            if not (15.0 <= duration <= 60.0):
                print(f"❌ FAIL (Length Error): {clip_name} ({duration} sec)")
                length_passed = False
            else:
                print(f"✅ PASS (Length OK): {clip_name} ({duration} sec)")
    
    if not clips:
        print("❌ FAIL: No rendered clips found in the directory.")
        length_passed = False

    # 2. Subtitle Presence Check
    subs = [f for f in os.listdir(SUBS_DIR) if f.endswith(('.srt', '.vtt'))] if os.path.exists(SUBS_DIR) else []
    if len(subs) >= len(clips) and len(clips) > 0:
        print(f"✅ PASS (Subtitles Found): {len(subs)} subtitle files detected.")
    else:
        print(f"❌ FAIL (Missing Subtitles): Found {len(subs)} subtitle files for {len(clips)} clips.")

    # 3. Quality Report & Hook Score Check
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
            # Assuming the QA report contains a key named 'average_hook_score'
            hook_score = qa_data.get("average_hook_score", 0)
            if hook_score >= 60:
                print(f"✅ PASS (Hook Score): {hook_score}/100 (Target: >= 60)")
            else:
                print(f"❌ FAIL (Low Hook Score): {hook_score}/100")
    else:
        print("❌ FAIL: quality_report.json file was not generated.")

    print("\n--- ✅ Evaluation Complete ---")

if __name__ == "__main__":
    evaluate_pipeline()

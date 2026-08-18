import os
import json
import cv2

OUTPUT_DIR = "output/"
CLIPS_DIR = os.path.join(OUTPUT_DIR, "formatted")
SUBS_DIR = os.path.join(OUTPUT_DIR, "subtitles")
REPORT_FILE = os.path.join(OUTPUT_DIR, "quality_report.json")

def get_video_duration(video_path):
    # OpenCV를 사용하여 영상 길이를 안전하게 추출합니다.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return duration

def evaluate_pipeline():
    print("--- 📊 Starting Pipeline Automated QA Check ---\n")
    
    # 1. Output Length Check (Must be 15~60 seconds)
    clips = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4")] if os.path.exists(CLIPS_DIR) else []
    length_passed = True
    for clip_name in clips:
        clip_path = os.path.join(CLIPS_DIR, clip_name)
        duration = get_video_duration(clip_path)
        
        if not (15.0 <= duration <= 60.0):
            print(f"❌ FAIL (Length Error): {clip_name} ({duration:.2f} sec)")
            length_passed = False
        else:
            print(f"✅ PASS (Length OK): {clip_name} ({duration:.2f} sec)")
    
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
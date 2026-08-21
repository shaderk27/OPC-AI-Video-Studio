import os
import json
import glob
from moviepy.editor import VideoFileClip

def get_latest_run_dir():
    search_pattern = os.path.join("output", "*", "*")
    folders = [f for f in glob.glob(search_pattern) if os.path.isdir(f)]
    
    if not folders:
        return None
        
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder

def evaluate_pipeline():
    print("--- 📊 Starting Pipeline Automated QA Check ---\n")
    
    latest_dir = get_latest_run_dir()
    if not latest_dir:
        print("❌ FAIL: No run directories found in 'output/'.")
        return
        
    print(f"🔎 Target Directory: {latest_dir}\n")
    
    final_dir = os.path.join(latest_dir, "final")
    target_dir = final_dir if os.path.exists(final_dir) else latest_dir
    
    report_file = os.path.join(latest_dir, "quality_report.json")
    if not os.path.exists(report_file):
        report_file = os.path.join(target_dir, "quality_report.json")

    clips = [f for f in os.listdir(target_dir) if f.endswith(".mp4")] if os.path.exists(target_dir) else []
    
    if not clips:
        print("❌ FAIL: No rendered clips (.mp4) found in the directory.")
    else:
        for clip_name in clips:
            clip_path = os.path.join(target_dir, clip_name)
            with VideoFileClip(clip_path) as video:
                duration = video.duration
                if not (15.0 <= duration <= 60.0):
                    print(f"❌ FAIL (Length Error): {clip_name} ({duration} sec)")
                else:
                    print(f"✅ PASS (Length OK): {clip_name} ({duration} sec)")
    
    if clips:
        sub_burned_clips = [f for f in clips if "sub-" in f.lower()]
        if len(sub_burned_clips) == len(clips):
            print(f"✅ PASS (Subtitles Burned): All {len(clips)} clips have subtitles successfully burned in.")
        else:
            missing_count = len(clips) - len(sub_burned_clips)
            print(f"❌ FAIL (Missing Subtitles): {missing_count} clips failed to burn subtitles.")

    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
            
            if "average_hook_score" not in qa_data:
                print("⚠️ SKIP (Hook Score): 'average_hook_score' not found in quality_report.json.")
            else:
                hook_score = qa_data["average_hook_score"]
                if hook_score >= 60:
                    print(f"✅ PASS (Hook Score): {hook_score}/100 (Target: >= 60)")
                else:
                    print(f"❌ FAIL (Low Hook Score): {hook_score}/100")
    else:
        print("❌ FAIL: quality_report.json file was not generated.")

    print("\n--- ✅ Evaluation Complete ---")

if __name__ == "__main__":
    evaluate_pipeline()

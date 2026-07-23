from pathlib import Path

raw_data_path = Path(r"D:\SHARED\PhD\data\speech_imagery_recordings")
old_spellitout_path = Path(r"D:\SHARED\PhD\data\Spell-IT-OUT\data\CortecchiaOvert2025")

subjects = sorted([subject.name for subject in raw_data_path.iterdir() if subject.is_dir()])
for subjno, subject in enumerate(subjects):
    subj_path = raw_data_path / subject
    overt_runs = sorted(list(subj_path.rglob("*overt.gdf")))
    
    new_subj_id = f"sub-{subjno+1:02d}"
    print(f"Subject: {new_subj_id} ({subject})")
    print(f"  Runs:")
    for run_no,orig_gdf_path in enumerate(overt_runs):
        run_id = orig_gdf_path.stem.split(".gdf")[0]
        old_annot_path = old_spellitout_path / subject / 'annotations' / f"{run_id.replace('.','_')}_aligned_vad_mfa_man_refined.TextGrid"
        old_audio_path = old_spellitout_path / subject / 'audio' / f"{run_id}.cropped.wav"
        
        # Missing subjects that I know: j9, l7, l9 
        new_audio_path = orig_gdf_path.parent / f"{run_id}.audio.wav"
        new_annot_path = orig_gdf_path.parent / f"{run_id}.annotations.TextGrid"

        print(f"    {run_no+1}): {run_id}, annotations: {old_annot_path.exists()}, audio: {old_audio_path.exists()}")

        # copy audio and annotation files to the new location
        if old_audio_path.exists() and old_annot_path.exists():
            new_audio_path.write_bytes(old_audio_path.read_bytes())
            new_annot_path.write_bytes(old_annot_path.read_bytes())
        else:
            print(f"      Warning: Missing files for {run_id}. Skipping copy.")



        

        
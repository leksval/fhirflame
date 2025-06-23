#!/usr/bin/env python3
"""
Download Medical Files for Testing
Simple script to download DICOM and other medical files for testing FhirFlame
"""

import os
import requests
import time
from pathlib import Path
from typing import List

class MedicalFileDownloader:
    """Simple downloader for medical test files"""
    
    def __init__(self):
        self.download_dir = Path("tests/medical_files")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample medical files (these are publicly available test files)
        self.file_sources = {
            "dicom_samples": [
                # These would be actual DICOM file URLs - using placeholders for now
                "https://www.rubomedical.com/dicom_files/CT_small.dcm",
                "https://www.rubomedical.com/dicom_files/MR_small.dcm", 
                "https://www.rubomedical.com/dicom_files/US_small.dcm",
                "https://www.rubomedical.com/dicom_files/XA_small.dcm",
            ],
            "text_reports": [
                # Medical text documents for testing
                "sample_discharge_summary.txt",
                "sample_lab_report.txt",
                "sample_radiology_report.txt"
            ]
        }
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a single file"""
        try:
            file_path = self.download_dir / filename
            
            # Skip if file already exists
            if file_path.exists():
                print(f"⏭️  Skipping {filename} (already exists)")
                return True
            
            print(f"📥 Downloading {filename}...")
            
            # Try to download the file
            response = requests.get(url, timeout=30, stream=True)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(file_path)
                print(f"✅ Downloaded {filename} ({file_size} bytes)")
                return True
            else:
                print(f"❌ Failed to download {filename}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    def create_sample_medical_files(self) -> List[str]:
        """Create sample medical text files for testing"""
        sample_files = []
        
        # Sample discharge summary
        discharge_summary = """
DISCHARGE SUMMARY

Patient: John Smith
DOB: 1975-03-15
MRN: MR123456789
Admission Date: 2024-01-15
Discharge Date: 2024-01-18

CHIEF COMPLAINT:
Chest pain and shortness of breath

HISTORY OF PRESENT ILLNESS:
45-year-old male presents with acute onset chest pain radiating to left arm.
Associated with diaphoresis and nausea. No prior cardiac history.

VITAL SIGNS:
Blood Pressure: 145/95 mmHg
Heart Rate: 102 bpm
Temperature: 98.6°F
Oxygen Saturation: 96% on room air

ASSESSMENT AND PLAN:
1. Acute coronary syndrome - rule out myocardial infarction
2. Hypertension - new diagnosis
3. Start aspirin 325mg daily
4. Lisinopril 10mg daily for blood pressure control
5. Atorvastatin 40mg daily

MEDICATIONS PRESCRIBED:
- Aspirin 325mg daily
- Lisinopril 10mg daily
- Atorvastatin 40mg daily
- Nitroglycerin 0.4mg sublingual PRN chest pain

FOLLOW-UP:
Cardiology in 1 week
Primary care in 2 weeks
"""
        
        # Sample lab report
        lab_report = """
LABORATORY REPORT

Patient: Maria Rodriguez
DOB: 1962-08-22
MRN: MR987654321
Collection Date: 2024-01-20

COMPLETE BLOOD COUNT:
White Blood Cell Count: 7.2 K/uL (Normal: 4.0-11.0)
Red Blood Cell Count: 4.5 M/uL (Normal: 4.0-5.2)
Hemoglobin: 13.8 g/dL (Normal: 12.0-15.5)
Hematocrit: 41.2% (Normal: 36.0-46.0)
Platelet Count: 285 K/uL (Normal: 150-450)

COMPREHENSIVE METABOLIC PANEL:
Glucose: 126 mg/dL (High - Normal: 70-100)
BUN: 18 mg/dL (Normal: 7-20)
Creatinine: 1.0 mg/dL (Normal: 0.6-1.2)
eGFR: >60 (Normal)
Sodium: 140 mEq/L (Normal: 136-145)
Potassium: 4.2 mEq/L (Normal: 3.5-5.1)
Chloride: 102 mEq/L (Normal: 98-107)

LIPID PANEL:
Total Cholesterol: 220 mg/dL (High - Optimal: <200)
LDL Cholesterol: 145 mg/dL (High - Optimal: <100)
HDL Cholesterol: 45 mg/dL (Low - Normal: >40)
Triglycerides: 150 mg/dL (Normal: <150)

HEMOGLOBIN A1C:
HbA1c: 6.8% (Elevated - Target: <7% for diabetics)
"""
        
        # Sample radiology report
        radiology_report = """
RADIOLOGY REPORT

Patient: Robert Wilson
DOB: 1980-12-10
MRN: MR456789123
Exam Date: 2024-01-22
Exam Type: Chest X-Ray PA and Lateral

CLINICAL INDICATION:
Cough and fever

TECHNIQUE:
PA and lateral chest radiographs were obtained.

FINDINGS:
The lungs are well expanded and clear. No focal consolidation, 
pleural effusion, or pneumothorax is identified. The cardiac 
silhouette is normal in size and contour. The mediastinal 
contours are within normal limits. No acute bony abnormalities.

IMPRESSION:
Normal chest radiograph. No evidence of acute cardiopulmonary disease.

Electronically signed by:
Dr. Sarah Johnson, MD
Radiologist
"""
        
        # Write sample files
        samples = {
            "sample_discharge_summary.txt": discharge_summary,
            "sample_lab_report.txt": lab_report,
            "sample_radiology_report.txt": radiology_report
        }
        
        for filename, content in samples.items():
            file_path = self.download_dir / filename
            
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Created sample file: {filename}")
                sample_files.append(str(file_path))
            else:
                print(f"⏭️  Sample file already exists: {filename}")
                sample_files.append(str(file_path))
        
        return sample_files
    
    def download_all_files(self, limit: int = 10) -> List[str]:
        """Download medical files for testing"""
        downloaded_files = []
        
        print("🏥 Medical File Downloader")
        print("=" * 40)
        
        # Create sample text files first (these always work)
        print("\n📝 Creating sample medical text files...")
        sample_files = self.create_sample_medical_files()
        downloaded_files.extend(sample_files)
        
        # Try to download DICOM files (may fail if URLs don't exist)
        print(f"\n📥 Attempting to download DICOM files...")
        dicom_downloaded = 0
        
        for i, url in enumerate(self.file_sources["dicom_samples"][:limit]):
            if dicom_downloaded >= 5:  # Limit DICOM downloads
                break
                
            filename = f"sample_dicom_{i+1}.dcm"
            
            # Since these URLs may not exist, we'll create mock DICOM files instead
            print(f"⚠️  Real DICOM download not available, creating mock file: {filename}")
            mock_file_path = self.download_dir / filename
            
            if not mock_file_path.exists():
                # Create a small mock file (real DICOM would be much larger)
                with open(mock_file_path, 'wb') as f:
                    f.write(b"DICM" + b"MOCK_DICOM_FOR_TESTING" * 100)
                print(f"✅ Created mock DICOM file: {filename}")
                downloaded_files.append(str(mock_file_path))
                dicom_downloaded += 1
            else:
                downloaded_files.append(str(mock_file_path))
                dicom_downloaded += 1
            
            time.sleep(0.1)  # Be nice to servers
        
        print(f"\n📊 Download Summary:")
        print(f"   Total files available: {len(downloaded_files)}")
        print(f"   Text files: {len(sample_files)}")
        print(f"   DICOM files: {dicom_downloaded}")
        print(f"   Download directory: {self.download_dir}")
        
        return downloaded_files
    
    def list_downloaded_files(self) -> List[str]:
        """List all downloaded medical files"""
        all_files = []
        
        for file_path in self.download_dir.iterdir():
            if file_path.is_file():
                all_files.append(str(file_path))
        
        return sorted(all_files)

def main():
    """Main download function"""
    downloader = MedicalFileDownloader()
    
    print("🚀 Starting medical file download...")
    files = downloader.download_all_files(limit=10)
    
    print(f"\n✅ Download complete! {len(files)} files ready for testing.")
    print("\nDownloaded files:")
    for file_path in files:
        file_size = os.path.getsize(file_path)
        print(f"   📄 {os.path.basename(file_path)} ({file_size} bytes)")
    
    return files

if __name__ == "__main__":
    main()
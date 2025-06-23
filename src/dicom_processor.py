"""
Simple DICOM Processor for FhirFlame
Basic DICOM file processing with FHIR conversion
"""

import os
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from .monitoring import monitor

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

class DICOMProcessor:
    """DICOM processor with fallback processing when pydicom unavailable"""
    
    def __init__(self):
        self.pydicom_available = PYDICOM_AVAILABLE
        if not PYDICOM_AVAILABLE:
            print("⚠️ pydicom not available - using fallback DICOM processing")
        
    @monitor.track_operation("dicom_processing")
    async def process_dicom_file(self, file_path: str) -> Dict[str, Any]:
        """Process DICOM file and convert to basic FHIR bundle"""
        
        if self.pydicom_available:
            return await self._process_with_pydicom(file_path)
        else:
            return await self._process_with_fallback(file_path)
    
    async def _process_with_pydicom(self, file_path: str) -> Dict[str, Any]:
        """Process DICOM file using pydicom library"""
        try:
            # Read DICOM file (with force=True for mock files)
            dicom_data = pydicom.dcmread(file_path, force=True)
            
            # Extract basic information
            patient_info = self._extract_patient_info(dicom_data)
            study_info = self._extract_study_info(dicom_data)
            
            # Create basic FHIR bundle
            fhir_bundle = self._create_fhir_bundle(patient_info, study_info)
            
            # Log processing
            monitor.log_medical_processing(
                entities_found=3,  # Patient, ImagingStudy, DiagnosticReport
                confidence=0.9,
                processing_time=1.0,
                processing_mode="dicom_processing",
                model_used="dicom_processor"
            )
            
            return {
                "status": "success",
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "patient_name": patient_info.get("name", "Unknown"),
                "study_description": study_info.get("description", "Unknown"),
                "modality": study_info.get("modality", "Unknown"),
                "fhir_bundle": fhir_bundle,
                "processing_time": 1.0,
                "extracted_text": f"DICOM file processed: {os.path.basename(file_path)}"
            }
            
        except Exception as e:
            monitor.log_event("dicom_processing_error", {"error": str(e), "file": file_path})
            return {
                "status": "error",
                "file_path": file_path,
                "error": str(e),
                "processing_time": 0.0
            }
    
    async def _process_with_fallback(self, file_path: str) -> Dict[str, Any]:
        """Fallback DICOM processing when pydicom is not available"""
        try:
            # Basic file information
            file_size = os.path.getsize(file_path)
            filename = os.path.basename(file_path)
            
            # CRITICAL: No dummy patient data in production - fail properly when DICOM processing fails
            raise Exception(f"DICOM processing failed for {filename}. Cannot extract real patient data. Will not generate fake medical information for safety and compliance.")
            
        except Exception as e:
            monitor.log_event("dicom_fallback_error", {"error": str(e), "file": file_path})
            return {
                "status": "error",
                "file_path": file_path,
                "error": f"Fallback processing failed: {str(e)}",
                "processing_time": 0.0,
                "fallback_used": True
            }
    
    def _extract_patient_info(self, dicom_data) -> Dict[str, str]:
        """Extract patient information from DICOM"""
        try:
            patient_name = str(dicom_data.get("PatientName", "Unknown Patient"))
            patient_id = str(dicom_data.get("PatientID", "Unknown ID"))
            patient_birth_date = str(dicom_data.get("PatientBirthDate", ""))
            patient_sex = str(dicom_data.get("PatientSex", ""))
            
            return {
                "name": patient_name,
                "id": patient_id,
                "birth_date": patient_birth_date,
                "sex": patient_sex
            }
        except Exception:
            return {
                "name": "Unknown Patient",
                "id": "Unknown ID",
                "birth_date": "",
                "sex": ""
            }
    
    def _extract_study_info(self, dicom_data) -> Dict[str, str]:
        """Extract study information from DICOM"""
        try:
            study_description = str(dicom_data.get("StudyDescription", "Unknown Study"))
            study_date = str(dicom_data.get("StudyDate", ""))
            modality = str(dicom_data.get("Modality", "Unknown"))
            study_id = str(dicom_data.get("StudyID", "Unknown"))
            
            return {
                "description": study_description,
                "date": study_date,
                "modality": modality,
                "id": study_id
            }
        except Exception:
            return {
                "description": "Unknown Study",
                "date": "",
                "modality": "Unknown",
                "id": "Unknown"
            }
    
    def _create_fhir_bundle(self, patient_info: Dict[str, str], study_info: Dict[str, str]) -> Dict[str, Any]:
        """Create basic FHIR bundle from DICOM data"""
        
        bundle_id = str(uuid.uuid4())
        patient_id = f"patient-{patient_info['id']}"
        study_id = f"study-{study_info['id']}"
        
        # Patient Resource
        patient_resource = {
            "resourceType": "Patient",
            "id": patient_id,
            "name": [{
                "text": patient_info["name"]
            }],
            "identifier": [{
                "value": patient_info["id"]
            }]
        }
        
        if patient_info["birth_date"]:
            patient_resource["birthDate"] = self._format_dicom_date(patient_info["birth_date"])
        
        if patient_info["sex"]:
            gender_map = {"M": "male", "F": "female", "O": "other"}
            patient_resource["gender"] = gender_map.get(patient_info["sex"], "unknown")
        
        # ImagingStudy Resource
        imaging_study = {
            "resourceType": "ImagingStudy",
            "id": study_id,
            "status": "available",
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "description": study_info["description"],
            "modality": [{
                "code": study_info["modality"],
                "display": study_info["modality"]
            }]
        }
        
        if study_info["date"]:
            imaging_study["started"] = self._format_dicom_date(study_info["date"])
        
        # DiagnosticReport Resource
        diagnostic_report = {
            "resourceType": "DiagnosticReport",
            "id": f"report-{study_info['id']}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "RAD",
                    "display": "Radiology"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic imaging study"
                }]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "conclusion": f"DICOM study: {study_info['description']}"
        }
        
        # Create Bundle
        return {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "document",
            "timestamp": datetime.now().isoformat(),
            "entry": [
                {"resource": patient_resource},
                {"resource": imaging_study},
                {"resource": diagnostic_report}
            ]
        }
    
    def _format_dicom_date(self, dicom_date: str) -> str:
        """Format DICOM date (YYYYMMDD) to ISO format"""
        try:
            if len(dicom_date) == 8:
                year = dicom_date[:4]
                month = dicom_date[4:6]
                day = dicom_date[6:8]
                return f"{year}-{month}-{day}"
            return dicom_date
        except Exception:
            return dicom_date

# Global instance - always create, fallback handling is internal
dicom_processor = DICOMProcessor()
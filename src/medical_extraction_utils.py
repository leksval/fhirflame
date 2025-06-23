#!/usr/bin/env python3
"""
Shared Medical Extraction Utilities
Centralized medical entity extraction logic to ensure consistency across all processors
"""

import re
from typing import Dict, Any, List
import json

class MedicalExtractor:
    """Centralized medical entity extraction with consistent patterns"""
    
    def __init__(self):
        # Comprehensive medical conditions database
        self.conditions_patterns = [
            "hypertension", "diabetes", "diabetes mellitus", "type 2 diabetes", "type 1 diabetes",
            "pneumonia", "asthma", "copd", "chronic obstructive pulmonary disease",
            "depression", "anxiety", "arthritis", "rheumatoid arthritis", "osteoarthritis",
            "cancer", "stroke", "heart disease", "coronary artery disease", "myocardial infarction",
            "kidney disease", "chronic kidney disease", "liver disease", "hepatitis",
            "chest pain", "acute coronary syndrome", "angina", "atrial fibrillation",
            "congestive heart failure", "heart failure", "cardiomyopathy",
            "hyperlipidemia", "high cholesterol", "obesity", "metabolic syndrome"
        ]
        
        # Common medication patterns
        self.medication_patterns = [
            r"([a-zA-Z]+(?:pril|sartan|olol|pine|statin|formin|cillin))\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|units?)\s+(daily|twice daily|bid|tid|qid|once daily)",
            r"(aspirin|lisinopril|atorvastatin|metformin|insulin|warfarin|prednisone|omeprazole)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|units?)",
            r"([a-zA-Z]+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|units?)\s+(daily|twice daily|bid|tid|qid)"
        ]
        
        # Vital signs patterns
        self.vital_patterns = [
            (r"bp:?\s*(\d{2,3}/\d{2,3})", "Blood Pressure"),
            (r"blood pressure:?\s*(\d{2,3}/\d{2,3})", "Blood Pressure"),
            (r"hr:?\s*(\d{2,3})", "Heart Rate"),
            (r"heart rate:?\s*(\d{2,3})", "Heart Rate"),
            (r"temp:?\s*(\d{2,3}(?:\.\d)?)", "Temperature"),
            (r"temperature:?\s*(\d{2,3}(?:\.\d)?)", "Temperature"),
            (r"o2 sat:?\s*(\d{2,3}%)", "O2 Saturation"),
            (r"oxygen saturation:?\s*(\d{2,3}%)", "O2 Saturation")
        ]
        
        # Procedures keywords
        self.procedures_keywords = [
            "ecg", "ekg", "electrocardiogram", "x-ray", "ct scan", "mri", "ultrasound",
            "blood test", "lab work", "biopsy", "endoscopy", "colonoscopy",
            "surgery", "operation", "procedure", "catheterization", "angiography"
        ]
    
    def extract_all_entities(self, text: str, processing_mode: str = "standard") -> Dict[str, Any]:
        """
        Extract all medical entities from text using consistent patterns
        
        Args:
            text: Medical text to analyze
            processing_mode: Processing mode for confidence scoring
        
        Returns:
            Dictionary with all extracted entities
        """
        return {
            "patient_info": self.extract_patient_info(text),
            "date_of_birth": self.extract_date_of_birth(text),
            "conditions": self.extract_conditions(text),
            "medications": self.extract_medications(text),
            "vitals": self.extract_vitals(text),
            "procedures": self.extract_procedures(text),
            "confidence_score": self.calculate_confidence_score(text, processing_mode),
            "extraction_quality": self.assess_extraction_quality(text),
            "processing_mode": processing_mode
        }
    
    def extract_patient_info(self, text: str) -> str:
        """Extract patient information with consistent patterns"""
        text_lower = text.lower()
        
        # Enhanced patient name patterns
        patterns = [
            r"patient:\s*([^\n\r,]+)",
            r"name:\s*([^\n\r,]+)", 
            r"pt\.?\s*([^\n\r,]+)",
            r"mr\.?\s*([^\n\r,]+)",
            r"patient name:\s*([^\n\r,]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).strip().title()
                # Validate name quality
                if (len(name) > 2 and 
                    not any(word in name.lower() for word in ['unknown', 'patient', 'test', 'sample']) and
                    re.match(r'^[a-zA-Z\s]+$', name)):
                    return name
        
        return "Unknown Patient"
    
    def extract_date_of_birth(self, text: str) -> str:
        """Extract date of birth with multiple formats"""
        text_lower = text.lower()
        
        # DOB patterns
        dob_patterns = [
            r"dob:?\s*([^\n\r]+)",
            r"date of birth:?\s*([^\n\r]+)",
            r"born:?\s*([^\n\r]+)",
            r"birth date:?\s*([^\n\r]+)"
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text_lower)
            if match:
                dob = match.group(1).strip()
                # Basic date validation
                if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[a-zA-Z]+ \d{1,2}, \d{4}', dob):
                    return dob
        
        return "Not specified"
    
    def extract_conditions(self, text: str) -> List[str]:
        """Extract medical conditions with context"""
        text_lower = text.lower()
        found_conditions = []
        
        for condition in self.conditions_patterns:
            if condition in text_lower:
                # Get context around the condition
                condition_pattern = rf"([^\n\r]*{re.escape(condition)}[^\n\r]*)"
                context_match = re.search(condition_pattern, text_lower)
                if context_match:
                    context = context_match.group(1).strip().title()
                    if context not in found_conditions and len(context) > len(condition):
                        found_conditions.append(context)
                elif condition.title() not in found_conditions:
                    found_conditions.append(condition.title())
        
        return found_conditions[:5]  # Limit to top 5 for clarity
    
    def extract_medications(self, text: str) -> List[str]:
        """Extract medications with dosages using consistent patterns"""
        medications = []
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 3:
                    med_name = match.group(1).title()
                    dose = match.group(2)
                    unit = match.group(3).lower()
                    frequency = match.group(4) if len(match.groups()) >= 4 else ""
                    
                    full_med = f"{med_name} {dose}{unit} {frequency}".strip()
                    if full_med not in medications:
                        medications.append(full_med)
        
        return medications[:5]  # Limit to top 5
    
    def extract_vitals(self, text: str) -> List[str]:
        """Extract vital signs with consistent formatting"""
        vitals = []
        
        for pattern, vital_type in self.vital_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                vital_value = match.group(1)
                
                if vital_type == "Blood Pressure":
                    vitals.append(f"Blood Pressure: {vital_value}")
                elif vital_type == "Heart Rate":
                    vitals.append(f"Heart Rate: {vital_value} bpm")
                elif vital_type == "Temperature":
                    vitals.append(f"Temperature: {vital_value}°F")
                elif vital_type == "O2 Saturation":
                    vitals.append(f"O2 Saturation: {vital_value}")
        
        return vitals[:4]  # Limit to top 4
    
    def extract_procedures(self, text: str) -> List[str]:
        """Extract procedures with consistent naming"""
        procedures = []
        text_lower = text.lower()
        
        for procedure in self.procedures_keywords:
            if procedure in text_lower:
                procedures.append(procedure.title())
        
        return procedures[:3]  # Limit to top 3
    
    def calculate_confidence_score(self, text: str, processing_mode: str) -> float:
        """Calculate confidence score based on text quality and processing mode"""
        base_confidence = {
            "rule_based": 0.75,
            "ollama": 0.85,
            "modal": 0.94,
            "huggingface": 0.88,
            "standard": 0.80
        }
        
        confidence = base_confidence.get(processing_mode, 0.80)
        
        # Adjust based on text quality
        if len(text) > 500:
            confidence += 0.05
        if len(text) > 1000:
            confidence += 0.05
        
        # Check for medical keywords
        medical_keywords = ["patient", "diagnosis", "medication", "treatment", "clinical"]
        keyword_count = sum(1 for keyword in medical_keywords if keyword.lower() in text.lower())
        confidence += keyword_count * 0.02
        
        return min(0.98, confidence)
    
    def assess_extraction_quality(self, text: str) -> Dict[str, Any]:
        """Assess the quality of extraction based on text content"""
        # Extract basic entities for quality assessment
        patient = self.extract_patient_info(text)
        dob = self.extract_date_of_birth(text)
        conditions = self.extract_conditions(text)
        medications = self.extract_medications(text)
        vitals = self.extract_vitals(text)
        procedures = self.extract_procedures(text)
        
        return {
            "patient_identified": patient != "Unknown Patient",
            "dob_found": dob != "Not specified",
            "conditions_count": len(conditions),
            "medications_count": len(medications),
            "vitals_count": len(vitals),
            "procedures_count": len(procedures),
            "total_entities": len(conditions) + len(medications) + len(vitals) + len(procedures),
            "detailed_medications": sum(1 for med in medications if any(unit in med.lower() for unit in ['mg', 'g', 'ml'])),
            "has_vital_signs": len(vitals) > 0,
            "comprehensive_analysis": len(conditions) > 0 and len(medications) > 0
        }
    
    def count_entities(self, extracted_data: Dict[str, Any]) -> int:
        """Count total entities consistently across the system"""
        return (len(extracted_data.get("conditions", [])) + 
                len(extracted_data.get("medications", [])) + 
                len(extracted_data.get("vitals", [])) + 
                len(extracted_data.get("procedures", [])))
    
    def format_for_pydantic(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format extracted data for Pydantic model compatibility"""
        return {
            "patient": extracted_data.get("patient_info", "Unknown Patient"),
            "date_of_birth": extracted_data.get("date_of_birth", "Not specified"),
            "conditions": extracted_data.get("conditions", []),
            "medications": extracted_data.get("medications", []),
            "vitals": extracted_data.get("vitals", []),
            "procedures": extracted_data.get("procedures", []),
            "confidence_score": extracted_data.get("confidence_score", 0.80),
            "extraction_quality": extracted_data.get("extraction_quality", {}),
            "_processing_metadata": {
                "mode": extracted_data.get("processing_mode", "standard"),
                "total_entities": self.count_entities(extracted_data),
                "extraction_timestamp": "2025-06-06T12:00:00Z"
            }
        }

# Global instance for consistent usage across the system
medical_extractor = MedicalExtractor()

# Convenience functions for backward compatibility
def extract_medical_entities(text: str, processing_mode: str = "standard") -> Dict[str, Any]:
    """Extract medical entities using the shared extractor"""
    return medical_extractor.extract_all_entities(text, processing_mode)

def count_entities(extracted_data: Dict[str, Any]) -> int:
    """Count entities using the shared method"""
    return medical_extractor.count_entities(extracted_data)

def format_for_pydantic(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format for Pydantic using the shared method"""
    return medical_extractor.format_for_pydantic(extracted_data)

def calculate_quality_score(extracted_data: Dict[str, Any]) -> float:
    """Calculate quality score based on entity richness"""
    entity_count = count_entities(extracted_data)
    patient_found = bool(extracted_data.get("patient_info") and 
                        extracted_data.get("patient_info") != "Unknown Patient")
    
    base_score = 0.7
    entity_bonus = min(0.25, entity_count * 0.04)  # Up to 0.25 bonus for entities
    patient_bonus = 0.05 if patient_found else 0
    
    return min(0.98, base_score + entity_bonus + patient_bonus)

# Export main components
__all__ = [
    "MedicalExtractor", 
    "medical_extractor", 
    "extract_medical_entities", 
    "count_entities", 
    "format_for_pydantic",
    "calculate_quality_score"
]
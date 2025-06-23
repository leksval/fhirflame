#!/usr/bin/env python3
"""
Real Medical Files Testing
Batch test FhirFlame on real medical files with performance metrics
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.file_processor import local_processor
from src.fhir_validator import FhirValidator
from src.monitoring import monitor
from tests.download_medical_files import MedicalFileDownloader

# Try to import DICOM processor
try:
    from src.dicom_processor import dicom_processor
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    dicom_processor = None

class MedicalFileTestFramework:
    """Simple testing framework for medical files"""
    
    def __init__(self):
        self.fhir_validator = FhirValidator()
        self.downloader = MedicalFileDownloader()
        self.results = []
        
        # Performance targets from the plan
        self.targets = {
            'success_rate': 0.90,    # >90% success
            'processing_time': 5.0,  # <5 seconds per file
            'fhir_compliance': 0.95  # >95% compliance
        }
    
    def analyze_mistral_ocr_compatibility(self, file_path: str) -> Dict[str, Any]:
        """Analyze if file is compatible with Mistral OCR"""
        file_path_lower = file_path.lower()
        
        # Image files - fully compatible
        if file_path_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return {
                'compatible': True,
                'confidence': 'high',
                'reason': 'Direct image format - ideal for Mistral OCR',
                'preprocessing_needed': False
            }
        
        # DICOM files - compatible with preprocessing
        elif file_path_lower.endswith(('.dcm', '.dicom')):
            return {
                'compatible': True,
                'confidence': 'medium',
                'reason': 'DICOM contains images but needs pixel data extraction',
                'preprocessing_needed': True
            }
        
        # PDF files - compatible with conversion
        elif file_path_lower.endswith('.pdf'):
            return {
                'compatible': True,
                'confidence': 'medium', 
                'reason': 'PDF can be converted to images for OCR',
                'preprocessing_needed': True
            }
        
        # Text files - not compatible (no OCR needed)
        elif file_path_lower.endswith(('.txt', '.text')):
            return {
                'compatible': False,
                'confidence': 'n/a',
                'reason': 'Plain text files - no OCR needed, process directly',
                'preprocessing_needed': False
            }
        
        # Unknown files
        else:
            return {
                'compatible': False,
                'confidence': 'unknown',
                'reason': 'Unknown file type - cannot determine OCR compatibility',
                'preprocessing_needed': False
            }
    
    def classify_file(self, file_path: str) -> str:
        """Classify file type"""
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith(('.dcm', '.dicom')):
            return 'dicom'
        elif file_path_lower.endswith(('.txt', '.text')):
            return 'text'
        elif file_path_lower.endswith('.pdf'):
            return 'pdf'
        elif file_path_lower.endswith(('.jpg', '.jpeg', '.png')):
            return 'image'
        else:
            return 'unknown'
    
    async def process_text_file(self, file_path: str) -> Dict[str, Any]:
        """Process text/PDF/image file using existing processor"""
        try:
            start_time = time.time()
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert to bytes for processor
            content_bytes = content.encode('utf-8')
            
            # Process with local processor (may use Mistral OCR if enabled)
            result = await local_processor.process_document(
                document_bytes=content_bytes,
                user_id="test-user",
                filename=os.path.basename(file_path)
            )
            
            processing_time = time.time() - start_time
            
            # Validate FHIR bundle
            fhir_validation = self.fhir_validator.validate_fhir_bundle(result['fhir_bundle'])
            
            # Check Mistral OCR compatibility
            ocr_compatibility = self.analyze_mistral_ocr_compatibility(file_path)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'file_type': 'text',
                'processing_time': processing_time,
                'entities_found': result['entities_found'],
                'fhir_valid': fhir_validation['is_valid'],
                'fhir_compliance': fhir_validation['compliance_score'],
                'processor_used': result['processing_mode'],
                'mistral_ocr_compatible': ocr_compatibility['compatible'],
                'mistral_ocr_notes': ocr_compatibility['reason']
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'status': 'error',
                'file_path': file_path,
                'file_type': 'text',
                'processing_time': processing_time,
                'error': str(e)
            }
    
    async def process_dicom_file(self, file_path: str) -> Dict[str, Any]:
        """Process DICOM file using DICOM processor"""
        if not DICOM_AVAILABLE or not dicom_processor:
            return {
                'status': 'error',
                'file_path': file_path,
                'file_type': 'dicom',
                'processing_time': 0.0,
                'error': 'DICOM processor not available - install pydicom',
                'mistral_ocr_compatible': True,
                'mistral_ocr_notes': 'DICOM images are compatible but need preprocessing'
            }
        
        try:
            start_time = time.time()
            
            # Process with DICOM processor
            result = await dicom_processor.process_dicom_file(file_path)
            
            processing_time = time.time() - start_time
            
            # Check Mistral OCR compatibility
            ocr_compatibility = self.analyze_mistral_ocr_compatibility(file_path)
            
            if result['status'] == 'success':
                # Validate FHIR bundle
                fhir_validation = self.fhir_validator.validate_fhir_bundle(result['fhir_bundle'])
                
                return {
                    'status': 'success',
                    'file_path': file_path,
                    'file_type': 'dicom',
                    'processing_time': processing_time,
                    'patient_name': result.get('patient_name', 'Unknown'),
                    'modality': result.get('modality', 'Unknown'),
                    'fhir_valid': fhir_validation['is_valid'],
                    'fhir_compliance': fhir_validation['compliance_score'],
                    'processor_used': 'dicom_processor',
                    'mistral_ocr_compatible': ocr_compatibility['compatible'],
                    'mistral_ocr_notes': ocr_compatibility['reason']
                }
            else:
                return {
                    'status': 'error',
                    'file_path': file_path,
                    'file_type': 'dicom',
                    'processing_time': processing_time,
                    'error': result.get('error', 'Unknown error'),
                    'mistral_ocr_compatible': ocr_compatibility['compatible'],
                    'mistral_ocr_notes': ocr_compatibility['reason']
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            ocr_compatibility = self.analyze_mistral_ocr_compatibility(file_path)
            return {
                'status': 'error',
                'file_path': file_path,
                'file_type': 'dicom',
                'processing_time': processing_time,
                'error': str(e),
                'mistral_ocr_compatible': ocr_compatibility['compatible'],
                'mistral_ocr_notes': ocr_compatibility['reason']
            }
    
    async def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single medical file"""
        file_type = self.classify_file(file_path)
        
        print(f"📄 Processing {os.path.basename(file_path)} ({file_type})...")
        
        if file_type == 'dicom':
            return await self.process_dicom_file(file_path)
        else:
            return await self.process_text_file(file_path)
    
    async def run_batch_test(self, file_limit: int = 20) -> Dict[str, Any]:
        """Run batch test on all medical files"""
        print("🏥 FhirFlame Medical File Batch Testing")
        print("=" * 50)
        
        # Download/prepare medical files
        print("📥 Preparing medical files...")
        available_files = self.downloader.download_all_files(limit=file_limit)
        
        if not available_files:
            print("❌ No medical files available for testing!")
            return {"error": "No files available"}
        
        print(f"📋 Found {len(available_files)} medical files to test")
        
        # Process each file
        start_time = time.time()
        self.results = []
        
        for i, file_path in enumerate(available_files, 1):
            print(f"\n[{i}/{len(available_files)}] ", end="")
            
            result = await self.process_single_file(file_path)
            self.results.append(result)
            
            # Show quick result
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            time_str = f"{result['processing_time']:.2f}s"
            ocr_note = "🔍OCR✅" if result.get('mistral_ocr_compatible') else "🔍OCR❌"
            print(f"{status_emoji} {time_str} {ocr_note}")
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self.generate_summary(total_time)
        
        print("\n" + "=" * 50)
        print("📊 BATCH TESTING RESULTS")
        print("=" * 50)
        
        return summary
    
    def generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate test summary and metrics"""
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Calculate metrics
        total_files = len(self.results)
        successful = [r for r in self.results if r['status'] == 'success']
        successful_count = len(successful)
        failed_count = total_files - successful_count
        
        success_rate = successful_count / total_files if total_files > 0 else 0
        
        # Processing time metrics
        processing_times = [r['processing_time'] for r in successful]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        
        # FHIR compliance metrics
        fhir_compliances = [r.get('fhir_compliance', 0) for r in successful]
        avg_fhir_compliance = sum(fhir_compliances) / len(fhir_compliances) if fhir_compliances else 0
        
        # Mistral OCR compatibility analysis
        ocr_compatible = [r for r in self.results if r.get('mistral_ocr_compatible', False)]
        ocr_incompatible = [r for r in self.results if not r.get('mistral_ocr_compatible', False)]
        
        # File type breakdown
        file_types = {}
        for result in self.results:
            file_type = result.get('file_type', 'unknown')
            if file_type not in file_types:
                file_types[file_type] = {'total': 0, 'successful': 0, 'ocr_compatible': 0}
            file_types[file_type]['total'] += 1
            if result['status'] == 'success':
                file_types[file_type]['successful'] += 1
            if result.get('mistral_ocr_compatible', False):
                file_types[file_type]['ocr_compatible'] += 1
        
        # Performance against targets
        meets_success_target = success_rate >= self.targets['success_rate']
        meets_time_target = avg_processing_time <= self.targets['processing_time']
        meets_compliance_target = avg_fhir_compliance >= self.targets['fhir_compliance']
        
        all_targets_met = meets_success_target and meets_time_target and meets_compliance_target
        
        # Print detailed results
        print(f"📋 Files Processed: {total_files}")
        print(f"✅ Successful: {successful_count} ({success_rate:.1%})")
        print(f"❌ Failed: {failed_count}")
        print(f"⏱️  Average Processing Time: {avg_processing_time:.2f}s")
        print(f"🔝 Maximum Processing Time: {max_processing_time:.2f}s")
        print(f"📊 Average FHIR Compliance: {avg_fhir_compliance:.1%}")
        print(f"🕐 Total Test Time: {total_time:.2f}s")
        
        print(f"\n🔍 Mistral OCR Compatibility Analysis:")
        print(f"   Compatible files: {len(ocr_compatible)}/{total_files} ({len(ocr_compatible)/total_files*100:.0f}%)")
        print(f"   Incompatible files: {len(ocr_incompatible)}/{total_files} ({len(ocr_incompatible)/total_files*100:.0f}%)")
        
        print(f"\n📂 File Type Breakdown:")
        for file_type, stats in file_types.items():
            success_pct = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
            ocr_pct = stats['ocr_compatible'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"   {file_type}: {stats['successful']}/{stats['total']} success ({success_pct:.0f}%) | OCR compatible: {stats['ocr_compatible']}/{stats['total']} ({ocr_pct:.0f}%)")
        
        print(f"\n🎯 Performance Targets:")
        print(f"   Success Rate: {success_rate:.1%} {'✅' if meets_success_target else '❌'} (target: {self.targets['success_rate']:.1%})")
        print(f"   Processing Time: {avg_processing_time:.2f}s {'✅' if meets_time_target else '❌'} (target: <{self.targets['processing_time']}s)")
        print(f"   FHIR Compliance: {avg_fhir_compliance:.1%} {'✅' if meets_compliance_target else '❌'} (target: {self.targets['fhir_compliance']:.1%})")
        
        print(f"\n🔍 Mistral OCR Data Type Support:")
        print(f"   ✅ Images (PNG, JPG): Direct compatibility")
        print(f"   ✅ DICOM files: Compatible with preprocessing") 
        print(f"   ✅ PDF files: Compatible with image conversion")
        print(f"   ❌ Plain text: No OCR needed (process directly)")
        
        print(f"\n🏆 Overall Result: {'✅ ALL TARGETS MET' if all_targets_met else '❌ Some targets missed'}")
        
        # Show errors if any
        errors = [r for r in self.results if r['status'] == 'error']
        if errors:
            print(f"\n❌ Errors ({len(errors)}):")
            for error in errors[:5]:  # Show first 5 errors
                filename = os.path.basename(error['file_path'])
                print(f"   {filename}: {error['error']}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")
        
        return {
            'total_files': total_files,
            'successful_count': successful_count,
            'failed_count': failed_count,
            'success_rate': success_rate,
            'avg_processing_time': avg_processing_time,
            'max_processing_time': max_processing_time,
            'avg_fhir_compliance': avg_fhir_compliance,
            'total_time': total_time,
            'file_types': file_types,
            'mistral_ocr_compatible_count': len(ocr_compatible),
            'mistral_ocr_incompatible_count': len(ocr_incompatible),
            'targets_met': {
                'success_rate': meets_success_target,
                'processing_time': meets_time_target,
                'fhir_compliance': meets_compliance_target,
                'all_targets': all_targets_met
            },
            'detailed_results': self.results
        }

async def main():
    """Main test function"""
    print(f"🕐 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if DICOM is available
    if DICOM_AVAILABLE:
        print("✅ DICOM processing available")
    else:
        print("⚠️  DICOM processing not available (install pydicom)")
    
    # Check Mistral OCR configuration
    mistral_enabled = os.getenv('USE_MISTRAL_FALLBACK', 'false').lower() == 'true'
    mistral_key = os.getenv('MISTRAL_API_KEY')
    
    print(f"🔍 Mistral OCR Status:")
    print(f"   Enabled: {mistral_enabled}")
    print(f"   API Key: {'✅ Set' if mistral_key else '❌ Missing'}")
    print(f"   Supported: Images, DICOM (preprocessed), PDF (converted)")
    print(f"   Not needed: Plain text files")
    
    # Run tests
    framework = MedicalFileTestFramework()
    
    try:
        results = await framework.run_batch_test(file_limit=15)
        
        if 'error' not in results:
            print(f"\n📋 Summary:")
            print(f"   {results['successful_count']}/{results['total_files']} files processed successfully")
            print(f"   {results['mistral_ocr_compatible_count']} files compatible with Mistral OCR")
            print(f"   Average time: {results['avg_processing_time']:.2f}s per file")
            print(f"   FHIR compliance: {results['avg_fhir_compliance']:.1%}")
            
        print(f"\n🎉 Medical file testing completed!")
        return 0
        
    except Exception as e:
        print(f"\n💥 Testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
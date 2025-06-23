"""
FhirFlame - Medical Document Intelligence Platform
CodeLlama 13B-instruct + RTX 4090 + MCP Server
"""

from .fhirflame_mcp_server import FhirFlameMCPServer
from .codellama_processor import CodeLlamaProcessor
from .fhir_validator import FhirValidator, ExtractedMedicalData, ProcessingMetadata
from .monitoring import FhirFlameMonitor, monitor, track_medical_processing, track_performance

__version__ = "0.1.0"
__all__ = [
    "FhirFlameMCPServer",
    "CodeLlamaProcessor",
    "FhirValidator",
    "ExtractedMedicalData",
    "ProcessingMetadata",
    "FhirFlameMonitor",
    "monitor",
    "track_medical_processing",
    "track_performance"
]
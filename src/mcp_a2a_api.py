#!/usr/bin/env python3
"""
FhirFlame MCP Server - Official MCP + A2A Standards Compliant API
Following official MCP protocol and FastAPI A2A best practices
Auth0 integration available for production (disabled for development)
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import os
import time
import httpx
# Optional Auth0 imports for production
try:
    from authlib.integrations.fastapi_oauth2 import AuthorizationCodeBearer
    AUTHLIB_AVAILABLE = True
except ImportError:
    AuthorizationCodeBearer = None
    AUTHLIB_AVAILABLE = False

from .fhirflame_mcp_server import FhirFlameMCPServer
from .monitoring import monitor

# Environment configuration
DEVELOPMENT_MODE = os.getenv("FHIRFLAME_DEV_MODE", "true").lower() == "true"
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "")

# Official MCP-compliant request/response models
class MCPToolRequest(BaseModel):
    """Official MCP tool request format"""
    name: str = Field(..., description="MCP tool name")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")

class MCPToolResponse(BaseModel):
    """Official MCP tool response format"""
    content: List[Dict[str, Any]] = Field(..., description="Response content")
    isError: bool = Field(default=False, description="Error flag")

# A2A-specific models following FastAPI standards
class ProcessDocumentRequest(BaseModel):
    document_content: str = Field(..., min_length=1, description="Medical document content")
    document_type: str = Field(default="clinical_note", description="Document type")
    extract_entities: bool = Field(default=True, description="Extract medical entities")
    generate_fhir: bool = Field(default=False, description="Generate FHIR bundle")

class ValidateFhirRequest(BaseModel):
    fhir_bundle: Dict[str, Any] = Field(..., description="FHIR bundle to validate")
    validation_level: str = Field(default="strict", pattern="^(strict|moderate|basic)$")

class A2AResponse(BaseModel):
    """A2A standard response format"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Initialize FastAPI with OpenAPI compliance
app = FastAPI(
    title="FhirFlame MCP A2A API",
    description="Official MCP-compliant API with A2A access to medical document processing",
    version="1.0.0",
    openapi_tags=[
        {"name": "mcp", "description": "Official MCP protocol endpoints"},
        {"name": "a2a", "description": "API-to-API endpoints"},
        {"name": "health", "description": "System health and monitoring"}
    ],
    docs_url="/docs" if DEVELOPMENT_MODE else None,  # Disable docs in production
    redoc_url="/redoc" if DEVELOPMENT_MODE else None
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if DEVELOPMENT_MODE else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize MCP server
mcp_server = FhirFlameMCPServer()
server_start_time = time.time()

# Authentication setup - Auth0 for production, simple key for development
security = HTTPBearer()

if not DEVELOPMENT_MODE and AUTH0_DOMAIN and AUTH0_AUDIENCE:
    # Production Auth0 setup
    auth0_scheme = AuthorizationCodeBearer(
        authorizationUrl=f"https://{AUTH0_DOMAIN}/authorize",
        tokenUrl=f"https://{AUTH0_DOMAIN}/oauth/token",
    )
    
    async def verify_token(token: str = Security(auth0_scheme)) -> Dict[str, Any]:
        """Verify Auth0 JWT token for production"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://{AUTH0_DOMAIN}/userinfo",
                    headers={"Authorization": f"Bearer {token}"}
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication credentials"
                    )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed"
            )
else:
    # Development mode - simple API key
    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
        """Simple API key verification for development"""
        if DEVELOPMENT_MODE:
            # In development, accept any token or skip auth entirely
            return "dev-user"
        
        expected_key = os.getenv("FHIRFLAME_API_KEY", "fhirflame-dev-key")
        if credentials.credentials != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return credentials.credentials

# Health check (no auth required)
@app.get("/health", tags=["health"])
async def health_check():
    """System health check - no authentication required"""
    start_time = time.time()
    
    try:
        health_data = {
            "status": "healthy",
            "service": "fhirflame-mcp-a2a",
            "mcp_server": "operational",
            "development_mode": DEVELOPMENT_MODE,
            "auth_provider": "auth0" if (AUTH0_DOMAIN and not DEVELOPMENT_MODE) else "dev-key",
            "uptime_seconds": time.time() - server_start_time,
            "version": "1.0.0"
        }
        
        # Log health check
        monitor.log_a2a_api_response(
            endpoint="/health",
            status_code=200,
            response_time=time.time() - start_time,
            success=True
        )
        
        return health_data
        
    except Exception as e:
        monitor.log_error_event(
            error_type="health_check_failure",
            error_message=str(e),
            stack_trace="",
            component="a2a_api_health",
            severity="warning"
        )
        raise HTTPException(status_code=500, detail="Health check failed")

# Official MCP Protocol Endpoints
@app.post("/mcp/tools/call", response_model=MCPToolResponse, tags=["mcp"])
async def mcp_call_tool(
    request: MCPToolRequest,
    user: Union[str, Dict[str, Any]] = Depends(verify_token)
) -> MCPToolResponse:
    """
    Official MCP protocol tool calling endpoint
    Follows MCP specification for tool invocation
    """
    start_time = time.time()
    user_id = user if isinstance(user, str) else user.get("sub", "unknown")
    input_size = len(str(request.arguments))
    
    # Log MCP request
    monitor.log_a2a_api_request(
        endpoint="/mcp/tools/call",
        method="POST",
        auth_method="bearer_token",
        request_size=input_size,
        user_id=user_id
    )
    
    try:
        with monitor.trace_operation("mcp_tool_call", {
            "tool_name": request.name,
            "user_id": user_id,
            "input_size": input_size
        }) as trace:
            result = await mcp_server.call_tool(request.name, request.arguments)
            processing_time = time.time() - start_time
            
            entities_found = 0
            if result.get("success") and "extraction_results" in result:
                entities_found = result["extraction_results"].get("entities_found", 0)
            
            # Log MCP tool execution
            monitor.log_mcp_tool(
                tool_name=request.name,
                success=result.get("success", True),
                processing_time=processing_time,
                input_size=input_size,
                entities_found=entities_found
            )
            
            # Log API response
            monitor.log_a2a_api_response(
                endpoint="/mcp/tools/call",
                status_code=200,
                response_time=processing_time,
                success=result.get("success", True),
                entities_processed=entities_found
            )
            
            # Convert to official MCP response format
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": str(result)
                }],
                isError=not result.get("success", True)
            )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log error
        monitor.log_error_event(
            error_type="mcp_tool_call_error",
            error_message=str(e),
            stack_trace="",
            component="mcp_api",
            severity="error"
        )
        
        monitor.log_a2a_api_response(
            endpoint="/mcp/tools/call",
            status_code=500,
            response_time=processing_time,
            success=False
        )
        
        return MCPToolResponse(
            content=[{
                "type": "error",
                "text": f"MCP tool call failed: {str(e)}"
            }],
            isError=True
        )

@app.get("/mcp/tools/list", tags=["mcp"])
async def mcp_list_tools(
    user: Union[str, Dict[str, Any]] = Depends(verify_token)
) -> Dict[str, Any]:
    """Official MCP tools listing endpoint"""
    try:
        tools = mcp_server.get_tools()
        return {
            "tools": tools,
            "protocol_version": "2024-11-05",  # Official MCP version
            "server_info": {
                "name": "fhirflame",
                "version": "1.0.0"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list MCP tools: {str(e)}"
        )

# A2A Endpoints for service-to-service integration
@app.post("/api/v1/process-document", response_model=A2AResponse, tags=["a2a"])
async def a2a_process_document(
    request: ProcessDocumentRequest,
    user: Union[str, Dict[str, Any]] = Depends(verify_token)
) -> A2AResponse:
    """
    A2A endpoint for medical document processing
    Follows RESTful API design patterns
    """
    start_time = time.time()
    user_id = user if isinstance(user, str) else user.get("sub", "unknown")
    text_length = len(request.document_content)
    
    # Log API request
    monitor.log_a2a_api_request(
        endpoint="/api/v1/process-document",
        method="POST",
        auth_method="bearer_token",
        request_size=text_length,
        user_id=user_id
    )
    
    # Log document processing start
    monitor.log_document_processing_start(
        document_type=request.document_type,
        text_length=text_length,
        extract_entities=request.extract_entities,
        generate_fhir=request.generate_fhir
    )
    
    try:
        with monitor.trace_document_workflow(request.document_type, text_length) as trace:
            result = await mcp_server.call_tool("process_medical_document", {
                "document_content": request.document_content,
                "document_type": request.document_type,
                "extract_entities": request.extract_entities,
                "generate_fhir": request.generate_fhir
            })
            
            processing_time = time.time() - start_time
            entities_found = 0
            fhir_generated = bool(result.get("fhir_bundle"))
            
            if result.get("success") and "extraction_results" in result:
                extraction = result["extraction_results"]
                entities_found = extraction.get("entities_found", 0)
                
                # Log medical entity extraction details
                if "medical_entities" in extraction:
                    medical = extraction["medical_entities"]
                    monitor.log_medical_entity_extraction(
                        conditions=len(medical.get("conditions", [])),
                        medications=len(medical.get("medications", [])),
                        vitals=len(medical.get("vital_signs", [])),
                        procedures=0,  # Not extracted yet
                        patient_info_found=bool(extraction.get("patient_info")),
                        confidence=extraction.get("confidence_score", 0.0)
                    )
            
            # Log document processing completion
            monitor.log_document_processing_complete(
                success=result.get("success", True),
                processing_time=processing_time,
                entities_found=entities_found,
                fhir_generated=fhir_generated,
                quality_score=result.get("extraction_results", {}).get("confidence_score", 0.0)
            )
            
            # Log API response
            monitor.log_a2a_api_response(
                endpoint="/api/v1/process-document",
                status_code=200,
                response_time=processing_time,
                success=result.get("success", True),
                entities_processed=entities_found
            )
            
            return A2AResponse(
                success=result.get("success", True),
                data=result,
                metadata={
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "user_id": user_id,
                    "api_version": "v1",
                    "endpoint": "process-document",
                    "entities_found": entities_found
                }
            )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log error
        monitor.log_error_event(
            error_type="document_processing_error",
            error_message=str(e),
            stack_trace="",
            component="a2a_process_document",
            severity="error"
        )
        
        # Log document processing failure
        monitor.log_document_processing_complete(
            success=False,
            processing_time=processing_time,
            entities_found=0,
            fhir_generated=False,
            quality_score=0.0
        )
        
        monitor.log_a2a_api_response(
            endpoint="/api/v1/process-document",
            status_code=500,
            response_time=processing_time,
            success=False
        )
        
        return A2AResponse(
            success=False,
            error=str(e),
            metadata={
                "processing_time": processing_time,
                "timestamp": time.time(),
                "endpoint": "process-document",
                "user_id": user_id
            }
        )

@app.post("/api/v1/validate-fhir", response_model=A2AResponse, tags=["a2a"])
async def a2a_validate_fhir(
    request: ValidateFhirRequest,
    user: Union[str, Dict[str, Any]] = Depends(verify_token)
) -> A2AResponse:
    """A2A endpoint for FHIR bundle validation"""
    start_time = time.time()
    
    try:
        result = await mcp_server.call_tool("validate_fhir_bundle", {
            "fhir_bundle": request.fhir_bundle,
            "validation_level": request.validation_level
        })
        
        return A2AResponse(
            success=result.get("success", True),
            data=result,
            metadata={
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "user_id": user if isinstance(user, str) else user.get("sub", "unknown"),
                "api_version": "v1",
                "endpoint": "validate-fhir"
            }
        )
        
    except Exception as e:
        return A2AResponse(
            success=False,
            error=str(e),
            metadata={
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "endpoint": "validate-fhir"
            }
        )

# OpenAPI specification endpoint
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi():
    """Get OpenAPI specification for API integration"""
    if not DEVELOPMENT_MODE:
        raise HTTPException(status_code=404, detail="Not found")
    return app.openapi()

# Root endpoint
@app.get("/")
async def root():
    """API root with service information"""
    return {
        "service": "FhirFlame MCP A2A API",
        "version": "1.0.0",
        "protocols": ["MCP", "REST A2A"],
        "development_mode": DEVELOPMENT_MODE,
        "authentication": {
            "provider": "auth0" if (AUTH0_DOMAIN and not DEVELOPMENT_MODE) else "api-key",
            "development_bypass": DEVELOPMENT_MODE
        },
        "endpoints": {
            "mcp": ["/mcp/tools/call", "/mcp/tools/list"],
            "a2a": ["/api/v1/process-document", "/api/v1/validate-fhir"],
            "health": ["/health"]
        },
        "documentation": "/docs" if DEVELOPMENT_MODE else "disabled"
    }

if __name__ == "__main__":
    import uvicorn
    
    print(f"🚀 Starting FhirFlame MCP A2A API")
    print(f"📋 Development mode: {DEVELOPMENT_MODE}")
    print(f"🔐 Auth provider: {'Auth0' if (AUTH0_DOMAIN and not DEVELOPMENT_MODE) else 'Dev API Key'}")
    print(f"📖 Documentation: {'/docs' if DEVELOPMENT_MODE else 'disabled'}")
    
    uvicorn.run(
        "mcp_a2a_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=DEVELOPMENT_MODE,
        log_level="info"
    )
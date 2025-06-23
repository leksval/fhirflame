#!/usr/bin/env python3
"""
FhirFlame PostgreSQL Database Manager
Handles persistent storage for job tracking, processing history, and system metrics
Uses the existing PostgreSQL database from the Langfuse infrastructure
"""

import psycopg2
import psycopg2.extras
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class DatabaseManager:
    """
    PostgreSQL database manager for FhirFlame job tracking and processing history
    Connects to the existing langfuse-db PostgreSQL instance
    """
    
    import os

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'langfuse-db'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'langfuse'),
            'user': os.getenv('DB_USER', 'langfuse'),
            'password': os.getenv('DB_PASSWORD', 'langfuse')
        }
        self.init_database()
    
    import sqlite3
    import os

    def get_connection(self):
        """Get PostgreSQL connection with proper configuration, fallback to SQLite"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            # Fallback to SQLite
            try:
                sqlite_path = os.getenv('SQLITE_DB_PATH', 'fhirflame_fallback.db')
                conn = sqlite3.connect(sqlite_path)
                print(f"✅ Connected to SQLite fallback database at {sqlite_path}")
                return conn
            except Exception as e2:
                print(f"❌ SQLite fallback connection failed: {e2}")
                raise e
    def init_database(self):
        """Initialize database schema with proper tables and indexes"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create fhirflame schema if not exists
            cursor.execute('CREATE SCHEMA IF NOT EXISTS fhirflame')
            
            # Create jobs table with comprehensive tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fhirflame.jobs (
                    id VARCHAR(255) PRIMARY KEY,
                    job_type VARCHAR(50) NOT NULL,
                    name TEXT NOT NULL,
                    text_input TEXT,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    provider_used VARCHAR(50),
                    success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    processing_time VARCHAR(50),
                    entities_found INTEGER,
                    error_message TEXT,
                    result_data JSONB,
                    file_path TEXT,
                    batch_id VARCHAR(255),
                    workflow_type VARCHAR(50)
                )
            ''')
            
            # Create batch jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fhirflame.batch_jobs (
                    id VARCHAR(255) PRIMARY KEY,
                    workflow_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    batch_size INTEGER DEFAULT 0,
                    processed_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failed_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fhirflame_jobs_status ON fhirflame.jobs(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fhirflame_jobs_created_at ON fhirflame.jobs(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fhirflame_jobs_job_type ON fhirflame.jobs(job_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fhirflame_batch_jobs_status ON fhirflame.batch_jobs(status)')
            
            # Create trigger for updated_at auto-update
            cursor.execute('''
                CREATE OR REPLACE FUNCTION fhirflame.update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql'
            ''')
            
            cursor.execute('''
                DROP TRIGGER IF EXISTS update_fhirflame_jobs_updated_at ON fhirflame.jobs
            ''')
            
            cursor.execute('''
                CREATE TRIGGER update_fhirflame_jobs_updated_at 
                BEFORE UPDATE ON fhirflame.jobs 
                FOR EACH ROW 
                EXECUTE FUNCTION fhirflame.update_updated_at_column()
            ''')
            
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ PostgreSQL database initialized with fhirflame schema")
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            # Don't raise - allow app to continue with in-memory fallback
    
    def add_job(self, job_data: Dict[str, Any]) -> bool:
        """Add new job to PostgreSQL database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Ensure required fields
            job_id = job_data.get('id', f"job_{int(time.time())}")
            job_type = job_data.get('job_type', 'text')
            name = job_data.get('name', 'Unknown Job')
            status = job_data.get('status', 'pending')
            
            cursor.execute('''
                INSERT INTO fhirflame.jobs (
                    id, job_type, name, text_input, status, provider_used,
                    success, processing_time, entities_found, error_message,
                    result_data, file_path, batch_id, workflow_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                job_id,
                job_type,
                name,
                job_data.get('text_input'),
                status,
                job_data.get('provider_used'),
                job_data.get('success'),
                job_data.get('processing_time'),
                job_data.get('entities_found'),
                job_data.get('error_message'),
                json.dumps(job_data.get('result_data')) if job_data.get('result_data') else None,
                job_data.get('file_path'),
                job_data.get('batch_id'),
                job_data.get('workflow_type')
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ Job added to PostgreSQL database: {job_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add job to PostgreSQL database: {e}")
            return False
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing job in PostgreSQL database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build update query dynamically
            update_fields = []
            values = []
            
            for field, value in updates.items():
                if field in ['status', 'provider_used', 'success', 'processing_time', 
                           'entities_found', 'error_message', 'result_data', 'completed_at']:
                    update_fields.append(f"{field} = %s")
                    if field == 'result_data' and value is not None:
                        values.append(json.dumps(value))
                    else:
                        values.append(value)
            
            if update_fields:
                values.append(job_id)
                
                query = f"UPDATE fhirflame.jobs SET {', '.join(update_fields)} WHERE id = %s"
                cursor.execute(query, values)
                
                conn.commit()
                cursor.close()
                conn.close()
                print(f"✅ Job updated in PostgreSQL database: {job_id}")
                return True
            
        except Exception as e:
            print(f"❌ Failed to update job in PostgreSQL database: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get specific job from PostgreSQL database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("SELECT * FROM fhirflame.jobs WHERE id = %s", (job_id,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                job_data = dict(row)
                if job_data.get('result_data'):
                    try:
                        job_data['result_data'] = json.loads(job_data['result_data'])
                    except:
                        pass
                return job_data
            return None
            
        except Exception as e:
            print(f"❌ Failed to get job from PostgreSQL database: {e}")
            return None
    
    def get_jobs_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent jobs for UI display"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute('''
                SELECT * FROM fhirflame.jobs 
                ORDER BY created_at DESC 
                LIMIT %s
            ''', (limit,))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            jobs = []
            for row in rows:
                job_data = dict(row)
                if job_data.get('result_data'):
                    try:
                        job_data['result_data'] = json.loads(job_data['result_data'])
                    except:
                        pass
                jobs.append(job_data)
            
            print(f"✅ Retrieved {len(jobs)} jobs from PostgreSQL database")
            return jobs
            
        except Exception as e:
            print(f"❌ Failed to get jobs history from PostgreSQL: {e}")
            return []
    
    def get_dashboard_metrics(self) -> Dict[str, int]:
        """Get dashboard metrics from PostgreSQL database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get total jobs
            cursor.execute("SELECT COUNT(*) FROM fhirflame.jobs")
            total_jobs = cursor.fetchone()[0]
            
            # Get completed jobs
            cursor.execute("SELECT COUNT(*) FROM fhirflame.jobs WHERE status = 'completed'")
            completed_jobs = cursor.fetchone()[0]
            
            # Get successful jobs
            cursor.execute("SELECT COUNT(*) FROM fhirflame.jobs WHERE success = true")
            successful_jobs = cursor.fetchone()[0]
            
            # Get failed jobs
            cursor.execute("SELECT COUNT(*) FROM fhirflame.jobs WHERE success = false")
            failed_jobs = cursor.fetchone()[0]
            
            # Get active jobs
            cursor.execute("SELECT COUNT(*) FROM fhirflame.jobs WHERE status IN ('pending', 'processing')")
            active_jobs = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            metrics = {
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'successful_jobs': successful_jobs,
                'failed_jobs': failed_jobs,
                'active_jobs': active_jobs
            }
            
            print(f"✅ Retrieved dashboard metrics from PostgreSQL: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"❌ Failed to get dashboard metrics from PostgreSQL: {e}")
            return {
                'total_jobs': 0,
                'completed_jobs': 0,
                'successful_jobs': 0,
                'failed_jobs': 0,
                'active_jobs': 0
            }
    
    def add_batch_job(self, batch_data: Dict[str, Any]) -> bool:
        """Add batch job to PostgreSQL database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            batch_id = batch_data.get('id', f"batch_{int(time.time())}")
            
            cursor.execute('''
                INSERT INTO fhirflame.batch_jobs (
                    id, workflow_type, status, batch_size, processed_count,
                    success_count, failed_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    processed_count = EXCLUDED.processed_count,
                    success_count = EXCLUDED.success_count,
                    failed_count = EXCLUDED.failed_count,
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                batch_id,
                batch_data.get('workflow_type', 'unknown'),
                batch_data.get('status', 'pending'),
                batch_data.get('batch_size', 0),
                batch_data.get('processed_count', 0),
                batch_data.get('success_count', 0),
                batch_data.get('failed_count', 0)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ Batch job added to PostgreSQL database: {batch_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add batch job to PostgreSQL database: {e}")
            return False

# Global database instance
db_manager = DatabaseManager()

def get_db_connection():
    """Backward compatibility function"""
    return db_manager.get_connection()
def clear_all_jobs():
    """Clear all jobs from the database - utility function for UI"""
    try:
        db_manager = DatabaseManager()
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # Clear both regular jobs and batch jobs
        cursor.execute("DELETE FROM fhirflame.jobs")
        cursor.execute("DELETE FROM fhirflame.batch_jobs")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ All jobs cleared from database")
        return True
        
    except Exception as e:
        print(f"❌ Failed to clear database: {e}")
        return False
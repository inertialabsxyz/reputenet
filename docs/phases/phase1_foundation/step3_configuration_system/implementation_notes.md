# Step 3: Configuration System - Implementation Notes

**Context:** Implementing type-safe configuration system with environment profiles
**Approach:** python-dotenv + Pydantic for simple, validated configuration management

---

## Implementation Strategy

### Configuration Architecture
Based on design decisions, implementing:
- **Pydantic BaseSettings** for type-safe configuration schema
- **Environment profiles** for different deployment contexts
- **Startup validation** to catch configuration errors early
- **Environment variable** approach for secret management

### File Structure
```
config/
├── __init__.py
├── settings.py              # Main configuration schema
├── base.py                  # Base configuration models
├── environments/
│   ├── __init__.py
│   ├── development.py       # Development environment
│   ├── testing.py          # Test environment
│   ├── production.py       # Production environment
│   └── mock.py             # Mock demonstration mode
└── validation.py           # Configuration validation utilities
```

---

## Core Configuration Implementation

### Base Configuration Models

#### config/base.py
```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pathlib import Path

class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    url: str = Field(default="sqlite:///reputenet.db", description="Database connection URL")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    timeout: float = Field(default=30.0, ge=1.0, description="Query timeout in seconds")

class APIConfig(BaseModel):
    """External API configuration."""
    mode: Literal["mock", "real"] = Field(default="mock", description="API operation mode")
    rate_limit: int = Field(default=100, ge=1, description="Requests per minute")
    timeout: float = Field(default=30.0, ge=1.0, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")

    # Future real API configuration
    ethereum_rpc_url: Optional[str] = Field(default=None, description="Ethereum RPC endpoint")
    web3_provider_timeout: float = Field(default=60.0, ge=1.0, description="Web3 provider timeout")

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    format: Literal["text", "json"] = Field(default="text", description="Log output format")
    file_path: Optional[Path] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size in bytes")  # 10MB
    backup_count: int = Field(default=5, ge=0, description="Number of backup log files")

class SecurityConfig(BaseModel):
    """Security and authentication configuration."""
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Application secret key")
    token_expire_hours: int = Field(default=24, ge=1, description="Token expiration in hours")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")

class AppConfig(BaseModel):
    """Core application configuration."""
    name: str = Field(default="ReputeNet", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment name")

    # Performance settings
    max_workers: int = Field(default=4, ge=1, le=32, description="Maximum worker processes")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
```

### Main Configuration Schema

#### config/settings.py
```python
from pydantic import BaseSettings, Field, validator
from typing import Optional
import os
from pathlib import Path

from .base import DatabaseConfig, APIConfig, LoggingConfig, SecurityConfig, AppConfig

class Settings(BaseSettings):
    """Main application configuration."""

    # Core application settings
    app: AppConfig = Field(default_factory=AppConfig)

    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    class Config:
        env_prefix = "REPUTENET_"
        env_nested_delimiter = "_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("app")
    def validate_app_config(cls, v):
        """Validate application configuration."""
        if v.debug and v.environment == "production":
            raise ValueError("Debug mode cannot be enabled in production")
        return v

    @validator("database")
    def validate_database_config(cls, v):
        """Validate database configuration."""
        if v.url.startswith("sqlite") and v.pool_size > 1:
            # SQLite doesn't use connection pooling
            v.pool_size = 1
        return v

    @validator("security")
    def validate_security_config(cls, v, values):
        """Validate security configuration."""
        app_config = values.get("app")
        if app_config and app_config.environment == "production":
            if v.secret_key == "dev-secret-key-change-in-production":
                raise ValueError("Secret key must be changed for production deployment")
            if "*" in v.cors_origins:
                raise ValueError("CORS origins must be restricted in production")
        return v

    def is_mock_mode(self) -> bool:
        """Check if application is running in mock mode."""
        return self.api.mode == "mock"

    def is_production(self) -> bool:
        """Check if application is running in production."""
        return self.app.environment == "production"

    def get_log_level(self) -> str:
        """Get the configured log level."""
        return self.logging.level
```

### Environment Profile Implementation

#### config/environments/development.py
```python
from ..settings import Settings
from ..base import DatabaseConfig, APIConfig, LoggingConfig, AppConfig

def get_development_settings() -> Settings:
    """Get development environment configuration."""
    return Settings(
        app=AppConfig(
            debug=True,
            environment="development"
        ),
        database=DatabaseConfig(
            url="sqlite:///dev_reputenet.db",
            echo=True  # Enable SQL logging in development
        ),
        api=APIConfig(
            mode="mock",
            rate_limit=1000  # Higher rate limit for development
        ),
        logging=LoggingConfig(
            level="DEBUG",
            format="text"
        )
    )
```

#### config/environments/testing.py
```python
from ..settings import Settings
from ..base import DatabaseConfig, APIConfig, LoggingConfig, AppConfig

def get_testing_settings() -> Settings:
    """Get testing environment configuration."""
    return Settings(
        app=AppConfig(
            debug=True,
            environment="testing"
        ),
        database=DatabaseConfig(
            url="sqlite:///:memory:",  # In-memory database for tests
            echo=False
        ),
        api=APIConfig(
            mode="mock",
            timeout=5.0,  # Faster timeouts for tests
            retry_attempts=1
        ),
        logging=LoggingConfig(
            level="WARNING",  # Reduce test noise
            format="text"
        )
    )
```

#### config/environments/production.py
```python
from ..settings import Settings
from ..base import DatabaseConfig, APIConfig, LoggingConfig, AppConfig, SecurityConfig

def get_production_settings() -> Settings:
    """Get production environment configuration."""
    return Settings(
        app=AppConfig(
            debug=False,
            environment="production"
        ),
        database=DatabaseConfig(
            # URL will be set via environment variable
            echo=False,
            pool_size=20,
            timeout=60.0
        ),
        api=APIConfig(
            mode="real",  # Production uses real APIs
            rate_limit=50,  # Conservative rate limiting
            timeout=30.0,
            retry_attempts=3
        ),
        logging=LoggingConfig(
            level="INFO",
            format="json",  # Structured logging for production
            file_path="/var/log/reputenet/app.log"
        ),
        security=SecurityConfig(
            # Secret key will be set via environment variable
            cors_origins=["https://reputenet.com"]  # Restrict CORS
        )
    )
```

#### config/environments/mock.py
```python
from ..settings import Settings
from ..base import DatabaseConfig, APIConfig, LoggingConfig, AppConfig

def get_mock_settings() -> Settings:
    """Get mock demonstration environment configuration."""
    return Settings(
        app=AppConfig(
            debug=False,
            environment="mock"
        ),
        database=DatabaseConfig(
            url="sqlite:///mock_reputenet.db",
            echo=False
        ),
        api=APIConfig(
            mode="mock",
            rate_limit=500,  # Higher limits for demos
            timeout=10.0
        ),
        logging=LoggingConfig(
            level="INFO",
            format="text"
        )
    )
```

### Configuration Loading and Validation

#### config/validation.py
```python
from typing import Optional
import os
from pathlib import Path

from .settings import Settings
from .environments.development import get_development_settings
from .environments.testing import get_testing_settings
from .environments.production import get_production_settings
from .environments.mock import get_mock_settings

def load_settings(environment: Optional[str] = None) -> Settings:
    """Load configuration based on environment."""

    if environment is None:
        environment = os.getenv("REPUTENET_ENVIRONMENT", "development")

    # Load environment-specific settings
    if environment == "development":
        settings = get_development_settings()
    elif environment == "testing":
        settings = get_testing_settings()
    elif environment == "production":
        settings = get_production_settings()
    elif environment == "mock":
        settings = get_mock_settings()
    else:
        raise ValueError(f"Unknown environment: {environment}")

    # Override with environment variables if present
    env_settings = Settings()

    # Merge settings (environment variables take precedence)
    return merge_settings(settings, env_settings)

def merge_settings(base: Settings, override: Settings) -> Settings:
    """Merge two settings objects, with override taking precedence."""
    # Simple implementation - in practice, you might want more sophisticated merging
    merged_data = base.dict()
    override_data = override.dict(exclude_unset=True)

    # Deep merge logic here
    merged_data.update(override_data)

    return Settings(**merged_data)

def validate_configuration(settings: Settings) -> list[str]:
    """Validate configuration and return list of issues."""
    issues = []

    # Check database accessibility
    if settings.database.url.startswith("postgresql://"):
        # Would check PostgreSQL connection in real implementation
        pass
    elif settings.database.url.startswith("sqlite://"):
        # Check SQLite file permissions
        if "://" in settings.database.url and settings.database.url != "sqlite:///:memory:":
            db_path = Path(settings.database.url.split("://")[1])
            if not db_path.parent.exists():
                issues.append(f"Database directory does not exist: {db_path.parent}")

    # Check log file directory
    if settings.logging.file_path:
        log_dir = settings.logging.file_path.parent
        if not log_dir.exists():
            issues.append(f"Log directory does not exist: {log_dir}")

    # Check production-specific requirements
    if settings.is_production():
        if settings.security.secret_key == "dev-secret-key-change-in-production":
            issues.append("Production secret key must be set")
        if not settings.database.url.startswith(("postgresql://", "mysql://")):
            issues.append("Production should use a proper database (PostgreSQL/MySQL)")

    return issues

def get_configuration_summary(settings: Settings) -> dict:
    """Get a summary of current configuration for logging/debugging."""
    return {
        "environment": settings.app.environment,
        "debug": settings.app.debug,
        "api_mode": settings.api.mode,
        "database_type": settings.database.url.split("://")[0],
        "log_level": settings.logging.level,
        "log_format": settings.logging.format
    }
```

#### config/__init__.py
```python
"""Configuration module for ReputeNet."""

from .settings import Settings
from .validation import load_settings, validate_configuration, get_configuration_summary

# Global configuration instance
_settings: Settings = None

def get_settings() -> Settings:
    """Get the current application settings."""
    global _settings
    if _settings is None:
        _settings = load_settings()

        # Validate configuration on first load
        issues = validate_configuration(_settings)
        if issues:
            raise RuntimeError(f"Configuration validation failed: {'; '.join(issues)}")

    return _settings

def reload_settings(environment: str = None) -> Settings:
    """Reload settings (useful for testing)."""
    global _settings
    _settings = load_settings(environment)
    return _settings

# Export commonly used items
__all__ = ["Settings", "get_settings", "reload_settings", "load_settings"]
```

---

## Environment Variable Configuration

### .env.example
```bash
# ReputeNet Configuration Example
# Copy to .env and customize for your environment

# Core Application Settings
REPUTENET_APP_NAME=ReputeNet
REPUTENET_APP_DEBUG=false
REPUTENET_APP_ENVIRONMENT=development

# Database Configuration
REPUTENET_DATABASE_URL=sqlite:///reputenet.db
REPUTENET_DATABASE_ECHO=false
REPUTENET_DATABASE_POOL_SIZE=10

# API Configuration
REPUTENET_API_MODE=mock
REPUTENET_API_RATE_LIMIT=100
REPUTENET_API_TIMEOUT=30.0

# For production with real APIs:
# REPUTENET_API_ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID

# Logging Configuration
REPUTENET_LOGGING_LEVEL=INFO
REPUTENET_LOGGING_FORMAT=text
# REPUTENET_LOGGING_FILE_PATH=/var/log/reputenet/app.log

# Security Configuration (CHANGE FOR PRODUCTION!)
REPUTENET_SECURITY_SECRET_KEY=dev-secret-key-change-in-production
REPUTENET_SECURITY_TOKEN_EXPIRE_HOURS=24
REPUTENET_SECURITY_CORS_ORIGINS=["*"]
```

### Environment Setup Scripts

#### scripts/setup_env.py
```python
#!/usr/bin/env python3
"""Environment setup script for ReputeNet."""

import os
import sys
from pathlib import Path
import secrets

def create_env_file(environment: str = "development"):
    """Create .env file for specified environment."""

    env_file = Path(".env")
    if env_file.exists():
        response = input(f".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Generate secure secret key
    secret_key = secrets.token_urlsafe(32)

    env_content = f"""# ReputeNet Environment Configuration
# Generated for {environment} environment

# Core Application
REPUTENET_APP_ENVIRONMENT={environment}
REPUTENET_APP_DEBUG={'true' if environment == 'development' else 'false'}

# Database
REPUTENET_DATABASE_URL=sqlite:///{environment}_reputenet.db

# API Configuration
REPUTENET_API_MODE=mock

# Security
REPUTENET_SECURITY_SECRET_KEY={secret_key}

# Logging
REPUTENET_LOGGING_LEVEL={'DEBUG' if environment == 'development' else 'INFO'}
"""

    env_file.write_text(env_content)
    print(f"Created .env file for {environment} environment")
    print(f"Secret key generated: {secret_key[:8]}...")

def validate_environment():
    """Validate current environment setup."""
    try:
        from config import get_settings, validate_configuration

        settings = get_settings()
        issues = validate_configuration(settings)

        if issues:
            print("❌ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Configuration is valid")
            return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReputeNet environment setup")
    parser.add_argument("--environment", "-e",
                       choices=["development", "testing", "production", "mock"],
                       default="development",
                       help="Environment to set up")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Validate current configuration")

    args = parser.parse_args()

    if args.validate:
        success = validate_environment()
        sys.exit(0 if success else 1)
    else:
        create_env_file(args.environment)
```

---

## Integration and Testing

### Usage in Application Code
```python
# In your application
from config import get_settings

settings = get_settings()

# Use configuration
database_url = settings.database.url
is_debug = settings.app.debug
api_mode = settings.api.mode

# Check modes
if settings.is_mock_mode():
    # Use mock implementations
    pass

if settings.is_production():
    # Production-specific logic
    pass
```

### Testing Configuration
```python
# tests/test_config.py
import pytest
from config import Settings, load_settings, validate_configuration

def test_development_config():
    """Test development configuration loading."""
    settings = load_settings("development")
    assert settings.app.debug is True
    assert settings.api.mode == "mock"
    assert settings.logging.level == "DEBUG"

def test_production_config_validation():
    """Test production configuration validation."""
    settings = load_settings("production")
    issues = validate_configuration(settings)

    # Should have issues with default secret key
    assert any("secret key" in issue.lower() for issue in issues)

def test_environment_variable_override():
    """Test environment variable override."""
    import os
    os.environ["REPUTENET_APP_DEBUG"] = "true"
    os.environ["REPUTENET_API_MODE"] = "real"

    settings = load_settings("production")
    # Environment variables should override profile defaults
    assert settings.app.debug is True
    assert settings.api.mode == "real"
```

---

## Documentation and Deployment

### Developer Setup Guide
1. Clone repository
2. Run `python scripts/setup_env.py --environment development`
3. Install dependencies: `uv install`
4. Validate setup: `python scripts/setup_env.py --validate`
5. Start development server

### Production Deployment
1. Set required environment variables:
   - `REPUTENET_SECURITY_SECRET_KEY`
   - `REPUTENET_DATABASE_URL`
   - `REPUTENET_API_ETHEREUM_RPC_URL`
2. Set `REPUTENET_APP_ENVIRONMENT=production`
3. Validate configuration before deployment
4. Monitor logs and configuration health

This implementation provides a robust, type-safe configuration system that supports the prototype development workflow while being ready for production deployment.
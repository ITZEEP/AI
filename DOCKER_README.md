# Docker Setup and Usage Guide

This guide provides comprehensive instructions for running the 잇집 AI OCR service using Docker.

## Prerequisites

- Docker Desktop installed (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- Google Cloud Vision API credentials
- At least 2GB of available RAM

## Quick Start

1. **Clone the repository and navigate to the project**
   ```bash
   cd D:\itzip\AI-develop
   ```

2. **Set up environment variables**
   ```bash
   # Copy the example environment file
   copy .env.example .env
   
   # Edit .env and add your API keys:
   # GOOGLE_APPLICATION_CREDENTIALS=credentials/google-vision-key.json
   # GOOGLE_API_KEY=your-google-api-key-here
   ```

3. **Place your Google Cloud credentials**
   - Download your service account JSON from Google Cloud Console
   - Save it as `credentials/google-vision-key.json`

4. **Build and run the service**
   ```bash
   docker-compose up -d
   ```

5. **Verify the service is running**
   ```bash
   # Check health endpoint
   curl http://localhost:8000/health
   
   # View API documentation
   # Open http://localhost:8000/docs in your browser
   ```

## Docker Architecture

### Dockerfile Features

- **Base Image**: Python 3.11 slim (Debian Bullseye)
- **Multi-stage Build**: Optimized for size and security
- **Dependencies**: All OCR, PDF processing, and AI libraries pre-installed
- **Health Check**: Built-in health monitoring
- **Resource Limits**: Memory and CPU constraints configured

### Docker Compose Configuration

The `docker-compose.yml` provides:

- **Service Name**: `ai-service`
- **Container Name**: `itzip-ai-service`
- **Network**: Isolated `itzip-network` for security
- **Volume Mounts**:
  - `./credentials`: Google Cloud credentials (read-only)
  - `./logs`: Application logs
  - `./data`: Vector store and documents
  - `./temp`: Temporary file processing
  - `./law_system`: Legal documents

## Volume Management

### Credentials Volume
```yaml
./credentials:/app/credentials:ro
```
- Mount your Google Cloud service account JSON here
- Read-only access for security
- Never commit credentials to Git

### Data Volume
```yaml
./data:/app/data
```
- Contains vector store for legal document analysis
- Persists between container restarts
- Initialize with `law_docs/` PDF files

### Logs Volume
```yaml
./logs:/app/logs
```
- Application logs persist outside container
- Useful for debugging and monitoring
- Rotate logs periodically to save space

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud JSON key | `/app/credentials/google-vision-key.json` |
| `GOOGLE_API_KEY` | Google Generative AI API key | Required |
| `PORT` | Service port | `8000` |
| `HOST` | Service host | `0.0.0.0` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FILE` | Log file path | `/app/logs/app.log` |

## Common Docker Commands

### Service Management
```bash
# Start the service
docker-compose up -d

# Stop the service
docker-compose down

# Restart the service
docker-compose restart

# View service status
docker-compose ps
```

### Debugging
```bash
# View logs
docker-compose logs -f

# View last 100 lines of logs
docker-compose logs --tail=100

# Access container shell
docker-compose exec ai-service bash

# Check resource usage
docker stats itzip-ai-service
```

### Maintenance
```bash
# Rebuild the image (after requirements change)
docker-compose build --no-cache

# Remove unused images
docker image prune

# Clean up everything (careful!)
docker-compose down -v --rmi all
```

## Resource Configuration

The service is configured with resource limits:

- **Memory**: 2GB limit, 1GB reserved
- **CPU**: 1.0 CPU limit, 0.5 CPU reserved

Adjust these in `docker-compose.yml` if needed:

```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## Health Monitoring

The service includes a health check that:
- Runs every 30 seconds
- Times out after 10 seconds
- Retries 3 times before marking unhealthy
- Waits 40 seconds before first check

Check health status:
```bash
docker inspect itzip-ai-service --format='{{.State.Health.Status}}'
```

## Troubleshooting

### Container won't start
1. Check logs: `docker-compose logs`
2. Verify credentials file exists: `ls credentials/`
3. Ensure ports are available: `netstat -an | findstr 8000`

### OCR not working
1. Verify Google Cloud credentials are valid
2. Check API quotas in Google Cloud Console
3. Review logs for authentication errors

### Out of memory errors
1. Increase Docker Desktop memory allocation
2. Adjust container memory limits in docker-compose.yml
3. Check for memory leaks in logs

### Vector store initialization fails
1. Ensure law_docs directory has PDF files
2. Check file permissions
3. Verify sufficient disk space

## Production Deployment

For production environments:

1. **Use Docker Swarm or Kubernetes** for orchestration
2. **Enable SSL/TLS** with a reverse proxy
3. **Set up monitoring** with Prometheus/Grafana
4. **Configure log aggregation** with ELK stack
5. **Use secrets management** instead of .env files
6. **Set up automated backups** for data volumes

## Security Best Practices

1. **Never expose credentials** in images or logs
2. **Use read-only mounts** where possible
3. **Run as non-root user** (already configured)
4. **Keep base images updated** regularly
5. **Scan images for vulnerabilities** with tools like Trivy
6. **Use network isolation** between services

## Performance Optimization

1. **Enable BuildKit** for faster builds:
   ```bash
   set DOCKER_BUILDKIT=1
   docker-compose build
   ```

2. **Use layer caching** effectively
3. **Minimize image size** with multi-stage builds
4. **Configure appropriate worker counts** based on CPU cores
5. **Monitor and adjust resource limits** based on usage

## Backup and Recovery

### Backup data volumes
```bash
# Backup vector store
docker run --rm -v ai-develop_data:/data -v %cd%:/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Backup logs
docker run --rm -v ai-develop_logs:/logs -v %cd%:/backup alpine tar czf /backup/logs-backup.tar.gz -C /logs .
```

### Restore from backup
```bash
# Restore vector store
docker run --rm -v ai-develop_data:/data -v %cd%:/backup alpine tar xzf /backup/data-backup.tar.gz -C /data

# Restore logs
docker run --rm -v ai-develop_logs:/logs -v %cd%:/backup alpine tar xzf /backup/logs-backup.tar.gz -C /logs
```

## Support

For issues or questions:
1. Check the logs first: `docker-compose logs`
2. Review this documentation
3. Check the main README.md for API usage
4. Submit issues to the project repository
import multiprocessing
import os

# Debugging
reload = False

# Logging
errorlog = "-"
loglevel = "info"
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Server Socket
bind = os.getenv("WEB_BIND", "0.0.0.0:8000")

# Worker Processes
workers = int(os.getenv("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
threads = int(os.getenv("PYTHON_MAX_THREADS", 1))
timeout = 60
keepalive = 2

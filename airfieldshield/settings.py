"""
Django settings for airfieldshield project.
Production hints included — uses environment variables when available.
"""

import os
from pathlib import Path
from datetime import timedelta

# BASE_DIR (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables (you can also use django-environ or python-decouple)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") in ("1", "True", "true", "yes")
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_crontab",


    # Third-party
    "rest_framework",
    "django_celery_results",   # optional, useful for storing celery task/results metadata

    # Project apps
    "core",
    "ingestion",
    "ai_models",
    "alerts",
    "dashboard",
    "utils",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "airfieldshield.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

CRONJOBS = [
    # run pipeline every 5 minutes
    ('*/5 * * * *', 'django.core.management.call_command', ['run_pipeline']),
]
CRONJOBS = [
    ('*/5 * * * *', 'django.core.management.call_command', ['run_pipeline'], '> /tmp/pipeline.log 2>&1'),
]



# Add near top or where you keep constants
AIRFIELDSHIELD = {
    "DEFAULT_THUNDERSTORM_THRESHOLD": 0.5,
    "DEFAULT_GALE_PROB_THRESHOLD": 0.5,
}

# If you plan to use Celery, add:
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "django-db"
# And in INSTALLED_APPS add 'django_celery_results' (you already have it in previous logs)


WSGI_APPLICATION = "airfieldshield.wsgi.application"

# Database
# Default: SQLite for quick start. For production, set POSTGRES_* env vars.
if os.environ.get("POSTGRES_DB"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql_psycopg2",
            "NAME": os.environ.get("POSTGRES_DB"),
            "USER": os.environ.get("POSTGRES_USER", "postgres"),
            "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
            "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
            "PORT": os.environ.get("POSTGRES_PORT", "5432"),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": str(BASE_DIR / "db.sqlite3"),
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization / timezone
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Static files & media
STATIC_URL = "/static/"
STATIC_ROOT = os.environ.get("STATIC_ROOT", str(BASE_DIR / "staticfiles"))
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "/media/"
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", str(BASE_DIR / "media"))

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Django REST Framework basic config
REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticatedOrReadOnly",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 50,
}

# Celery configuration (uses Redis broker by default)
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "django-db")  # use django_celery_results
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE

# If using django-celery-results:
if "django_celery_results" in INSTALLED_APPS:
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "django-db")

# Logging (simple console logger tuned for development)
LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "INFO")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {"format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"},
        "simple": {"format": "%(levelname)s %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "verbose"},
    },
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {
        "django": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
        "airfieldshield": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
    },
}

# Security recommendations for production (edit when deploying)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = os.environ.get("DJANGO_SESSION_COOKIE_SECURE", "0") in ("1", "True", "true")
CSRF_COOKIE_SECURE = os.environ.get("DJANGO_CSRF_COOKIE_SECURE", "0") in ("1", "True", "true")

# Custom config for the app
AIRFIELDSHIELD = {
    "DEFAULT_ALERT_HORIZONS_MIN": [30, 60, 180, 24 * 60],  # minutes: 30m, 1h, 3h, 24h
    "DEFAULT_THUNDERSTORM_THRESHOLD": float(os.environ.get("THUNDER_THRESHOLD", 0.5)),
    "DEFAULT_GALE_WIND_THRESHOLD_MS": float(os.environ.get("GALE_WIND_THRESHOLD_MS", 17.0)),  # ~61 km/h
}

import logging

from supabase import Client, create_client

from config.settings import settings

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    """Return the module-level Supabase client, creating it on first call."""
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        logger.info("supabase | client initialised | url=%s", settings.supabase_url)
    return _client

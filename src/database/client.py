import logging

from supabase import AsyncClient, Client, acreate_client, create_client

from config.settings import settings

logger = logging.getLogger(__name__)

# Async client — used by FastAPI route handlers and CRUD functions (await-able).
_async_client: AsyncClient | None = None

# Sync client — used by background tasks and Streamlit (which is synchronous).
_sync_client: Client | None = None


async def get_client() -> AsyncClient:
    """Return the module-level async Supabase client (service-role key).

    Uses the service-role key for server-side writes — bypasses RLS intentionally.
    Call this from FastAPI route handlers and async CRUD functions.
    """
    global _async_client
    if _async_client is None:
        _async_client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        logger.info("supabase | async client initialised | url=%s", settings.supabase_url)
    return _async_client


def get_anon_client() -> Client:
    """Return a synchronous read-only Supabase client (anon key).

    Uses the anon key so RLS read policies are enforced.
    Use this in the Streamlit dashboard for all SELECT queries.
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )
        logger.info("supabase | anon client initialised | url=%s", settings.supabase_url)
    return _sync_client

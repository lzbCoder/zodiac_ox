from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from loguru import logger
from config import LANGCHAIN_API_KEY, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT

otel_enabled: bool = True


class _ToggleableProcessor:
    """SpanProcessor wrapper whose underlying processor can be swapped at runtime.

    OTel's ``set_tracer_provider`` can only be called once (enforced by
    ``_TRACER_PROVIDER_SET_ONCE.do_once``).  To allow dynamic enable/disable
    we keep a *single* TracerProvider and swap the inner processor here.
    """

    def __init__(self):
        self._wrapped: SpanProcessor | None = None
        self._shutdown = False

    def set(self, processor: SpanProcessor | None):
        if self._wrapped is not None:
            self._wrapped.shutdown()
        self._wrapped = processor

    def on_start(self, span, parent_context=None):
        if self._wrapped is not None:
            self._wrapped.on_start(span, parent_context)

    def on_end(self, span):
        if self._wrapped is not None:
            self._wrapped.on_end(span)

    def _on_ending(self, span):
        """Called by SynchronousMultiSpanProcessor instead of on_end."""
        if self._wrapped is not None:
            fn = getattr(self._wrapped, "_on_ending", self._wrapped.on_end)
            fn(span)

    def shutdown(self):
        self._shutdown = True
        if self._wrapped is not None:
            self._wrapped.shutdown()
            self._wrapped = None

    def force_flush(self, timeout_millis=30000):
        if self._wrapped is not None:
            self._wrapped.force_flush(timeout_millis)


_toggle: _ToggleableProcessor | None = None


def is_otel_enabled() -> bool:
    return otel_enabled


def set_otel_enabled(enabled: bool):
    """Dynamically enable/disable span export. Does NOT call ``set_tracer_provider``."""
    global otel_enabled
    otel_enabled = enabled

    if _toggle is None:
        return

    if enabled:
        if not LANGCHAIN_API_KEY:
            logger.info("OTel: LANGCHAIN_API_KEY not set — cannot enable")
            return
        otlp_exporter = OTLPSpanExporter(
            endpoint=LANGCHAIN_ENDPOINT,
            headers={
                "x-api-key": LANGCHAIN_API_KEY,
                "Langsmith-Project": LANGCHAIN_PROJECT,
            },
        )
        _toggle.set(BatchSpanProcessor(otlp_exporter))
        logger.info(f"OTel + LangSmith enabled (project={LANGCHAIN_PROJECT}, endpoint={LANGCHAIN_ENDPOINT})")
    else:
        _toggle.set(None)
        logger.info("OTel monitoring disabled — span export stopped")


def init_otel_tracer():
    """Create the *single* TracerProvider with a toggleable processor.

    Must be called once at module level — before any ``trace.get_tracer()`` —
    so that the provider is in place for route instrumentation.  The exporter
    is activated later by ``apply_otel_config_from_db`` during lifespan.
    """
    global _toggle, otel_enabled

    provider = trace.get_tracer_provider()
    if provider is not None and not isinstance(provider, trace.ProxyTracerProvider):
        return

    if not LANGCHAIN_API_KEY:
        logger.info("OTel: LANGCHAIN_API_KEY not set — tracing disabled")
        return

    resource = Resource.create({SERVICE_NAME: LANGCHAIN_PROJECT})
    tracer_provider = TracerProvider(resource=resource)

    _toggle = _ToggleableProcessor()
    tracer_provider.add_span_processor(_toggle)
    trace.set_tracer_provider(tracer_provider)
    logger.info("OTel provider initialized (exporter not yet activated)")


async def apply_otel_config_from_db():
    """Called during lifespan to sync otel_enabled from DB after tables are ready."""
    global otel_enabled
    try:
        from sqlalchemy import select
        from database import async_session
        from models.system_config import SystemConfig

        async with async_session() as db:
            stmt = select(SystemConfig).where(SystemConfig.config_key == "otel.enabled")
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            if row:
                db_enabled = row.config_value.lower() in ("true", "1", "yes")
                if db_enabled != otel_enabled:
                    set_otel_enabled(db_enabled)
            else:
                if not otel_enabled:
                    set_otel_enabled(True)
    except Exception as e:
        logger.warning(f"OTel config load from DB failed, keeping current state: {e}")

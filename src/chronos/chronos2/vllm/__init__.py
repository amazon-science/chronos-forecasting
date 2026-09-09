"""vLLM Chronos-2 Time Series Forecasting Model Plugin.

This plugin registers the Chronos-2 model with vLLM's ModelRegistry,
allowing it to be used with vLLM's inference engine for time series forecasting.
"""


def register_chronos2_model() -> None:
    """Register Chronos-2 models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded
    through vLLM's plugin discovery mechanism.
    """
    try:
        from vllm.logger import init_logger
        from vllm.model_executor.models.registry import ModelRegistry

        logger = init_logger(__name__)

        ModelRegistry.register_model(
            "Chronos2Model",
            "chronos.chronos2.vllm.model:Chronos2ForForecasting",
        )

        logger.info("Successfully registered Chronos-2 model with vLLM")

    except Exception as e:
        from vllm.logger import init_logger

        logger = init_logger(__name__)
        logger.error(f"Failed to register Chronos-2 model: {e}")
        raise


def get_chronos2_io_processor():
    """
    Factory function for IOProcessor plugin registration.

    This function is called by vLLM when --io-processor-plugin is set to 'chronos2'.
    Returns the fully qualified name (module.Class format) as a string.
    """
    return "chronos.chronos2.vllm.io_processor.Chronos2IOProcessor"


__all__ = [
    "register_chronos2_model",
    "get_chronos2_io_processor",
]
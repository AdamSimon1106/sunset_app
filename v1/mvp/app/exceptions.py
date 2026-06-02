class ForecastError(Exception):
    """Base exception for forecast/prediction pipeline errors."""


class DataUnavailableError(ForecastError):
    """Raised when external data could not be fetched reliably."""


class ModelInputError(ForecastError):
    """Raised when the model input could not be assembled correctly."""


class PredictionError(ForecastError):
    """Raised when the prediction pipeline fails after input assembly."""

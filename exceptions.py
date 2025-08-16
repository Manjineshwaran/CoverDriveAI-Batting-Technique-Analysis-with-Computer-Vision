class CoverDriveAnalysisError(Exception):
    """Base exception class for all Cover Drive Analysis errors."""
    def __init__(self, message: str = "An error occurred in Cover Drive Analysis", details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message

class VideoProcessingError(CoverDriveAnalysisError):
    """Raised when there's an error during video processing."""
    def __init__(self, message: str = "Video processing error", details: str = None):
        super().__init__(f"Video Processing Error: {message}", details)

class PoseEstimationError(CoverDriveAnalysisError):
    """Raised when there's an error during pose estimation."""
    def __init__(self, message: str = "Pose estimation error", details: str = None):
        super().__init__(f"Pose Estimation Error: {message}", details)

class MetricCalculationError(CoverDriveAnalysisError):
    """Raised when there's an error during metric calculation."""
    def __init__(self, message: str = "Metric calculation error", details: str = None):
        super().__init__(f"Metric Calculation Error: {message}", details)

class FileHandlingError(CoverDriveAnalysisError):
    """Raised when there's an error during file operations."""
    def __init__(self, message: str = "File handling error", details: str = None, filepath: str = None):
        if filepath:
            message = f"{message}. File: {filepath}"
        super().__init__(f"File Handling Error: {message}", details)
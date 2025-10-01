# core/exceptions.py


class PhotoProcessingError(Exception):
    """Base class for all photo processing errors."""

    def __init__(
        self, message: str = "An error occurred while processing the photo"
    ) -> None:
        self.message = message
        super().__init__(self.message)


class NoFaceFoundError(PhotoProcessingError):
    """Error: no face found in the photo."""

    def __init__(
        self, message: str = "No face found in the photo. Please upload a valid photo."
    ) -> None:
        super().__init__(message)


class InvalidImageFormatError(PhotoProcessingError):
    """Error: invalid file format."""

    def __init__(self, message: str = "Supported formats: .jpg, .jpeg.") -> None:
        super().__init__(message)


class CompressionError(PhotoProcessingError):
    """Error: unable to compress image to required size."""

    def __init__(self, message="Unable to compress image to required size.") -> None:
        super().__init__(message)

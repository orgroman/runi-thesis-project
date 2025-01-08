def __getattr__(name):
    if name == 'NegationDetection':
        from .negation_detection import NegationDetection
        return NegationDetection
    raise AttributeError(f"module {__name__} has no attribute {name}")
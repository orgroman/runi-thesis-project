def create_model(**kwargs):
    provider_type = kwargs.pop("provider_type")
    if provider_type == "llamafile":
        from .llamafile import LLamaFileModel
        return LLamaFileModel(**kwargs)
    
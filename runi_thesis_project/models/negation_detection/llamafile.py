from aiohttp import ClientSession

class LLamaFileModel:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def create_model(cls, **kwargs):
        return cls(**kwargs)
                
    async def apredict_text(self, text):
        async with ClientSession() as session:
            async with session.get(self.url, params={'text': text}) as response:
                return bool(await response.json()['result'])

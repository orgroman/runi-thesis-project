from pydantic import BaseModel

class MyModel(BaseModel):
    id: str
    completion_window: str

if __name__ == "__main__":
    model = MyModel(id="123", completion_window="24h")
    print(model.model_dump())
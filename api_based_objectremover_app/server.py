from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name : int
    number : str

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "dev"}

@app.get("/dev")
def read_dev():
    return {"Hello": "day1"}

@app.post("/dev")
async def read_dev(payload: Item):
    dev = payload.name
    num = payload.number
    return dev

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0", port=8000, log_level="debug")

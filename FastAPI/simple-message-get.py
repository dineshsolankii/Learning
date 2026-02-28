# import fastapi
# app = fastapi.FastAPI()

from fastapi import FastAPI
app = FastAPI() # Fast api instance is "app"

@app.get("/")
def method_get():
    return {"message": "Hello World"}
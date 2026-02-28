# import fastapi
# app = fastapi.FastAPI()

from fastapi import FastAPI
app = FastAPI() # Fast api instance is "app"

@app.get("/")
async def method_get(): # async is used for handling multiple requests
    return {"message": "Hello World"}
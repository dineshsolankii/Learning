# import fastapi
# app = fastapi.FastAPI()

from fastapi import FastAPI
app = FastAPI() # Fast api instance is "app"

@app.get("/") # decorator
async def read_root(): # async is used for handling multiple requests
    return {"message": "Welcome to my FastAPI"}
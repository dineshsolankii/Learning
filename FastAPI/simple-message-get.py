# import fastapi
# app = fastapi.FastAPI()

from fastapi import FastAPI
app = FastAPI() # Fast api instance is "app"

@app.get("/") # decorator
async def read_root(): # async is used for handling multiple requests
    return {"message": "Welcome to my FastAPI"}

@app.get("/posts")
async def read_posts():
    return {"message": "This is your post"}

@app.post("/postrequest")
async def create_post():
    return {"message": "Post created"}
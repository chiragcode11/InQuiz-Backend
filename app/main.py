from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from database import close_database_connection
import uvicorn

app = FastAPI(title="AI Interviewer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "AI Interviewer API is running"}

@app.on_event("shutdown")
async def shutdown_event():
    close_database_connection()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

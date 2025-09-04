from pymongo import MongoClient
from pymongo.database import Database
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ai_interviewer")

client = MongoClient(MONGODB_URL)
database: Database = client[DATABASE_NAME]

def get_database():
    return database

def close_database_connection():
    client.close()

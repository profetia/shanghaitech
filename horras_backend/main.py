from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from horras_backend.api.v1 import router as api_router
from horras_backend.database import prisma

app = FastAPI(
    title="Horras Backend",
    description="Backend for Horras",
    version='0.1.0', 
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1", tags=["api"])

@app.on_event("startup")
async def startup():
    await prisma.connect()

@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from facebook_api import router

app = FastAPI()
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")



if __name__ == "__main__":
    import uvicorn
    print("Starting Facebook API Service...")
    # It's better to run using: uvicorn facebook_api:app --reload --port 8001
    uvicorn.run(app, host="0.0.0.0", port=6969)


    # Fetch Insights:
    # curl -X GET "${API_URL}/facebook/account/${ACCOUNT_ID}/insights?timeframe=7d&status_filter=active" -H "X-Facebook-Access-Token: ${ACCESS_TOKEN}"

    # Fetch Demographics:
    # curl -X GET "${API_URL}/facebook/ad/${AD_ID}/demographics?timeframe=30d" -H "X-Facebook-Access-Token: ${ACCESS_TOKEN}"

    # Fetch Thumbnail:
    # curl -X GET "${API_URL}/facebook/ad/${AD_ID}/thumbnail" -H "X-Facebook-Access-Token: ${ACCESS_TOKEN}"

    # Fetch Embed URL:
    # curl -X GET "${API_URL}/facebook/ad/${AD_ID}/embed_url" -H "X-Facebook-Access-Token: ${ACCESS_TOKEN}"

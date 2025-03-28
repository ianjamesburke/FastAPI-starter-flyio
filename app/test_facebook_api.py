import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = TestClient(app)

# Facebook test credentials from environment
FB_ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")
FB_ACCOUNT_ID = "749376029556624"  # Format: "act_123456789" or "123456789"
FB_AD_ID = "23859991584090431" # A real ad ID from your account

@pytest.mark.skipif(
    not all([FB_ACCESS_TOKEN, FB_ACCOUNT_ID, FB_AD_ID]),
    reason="Facebook API credentials not configured"
)
def test_get_ad_thumbnail_real():
    """Test getting thumbnail with real Facebook credentials"""
    response = client.get(
        f"/facebook/ad/{FB_AD_ID}/thumbnail",
        headers={"X-Facebook-Access-Token": FB_ACCESS_TOKEN}
    )
    assert response.status_code == 200
    assert "thumbnail_url" in response.json()
    thumbnail_url = response.json()["thumbnail_url"]
    assert thumbnail_url is None or thumbnail_url.startswith("http")

@pytest.mark.skipif(
    not all([FB_ACCESS_TOKEN, FB_ACCOUNT_ID, FB_AD_ID]),
    reason="Facebook API credentials not configured"
)
def test_get_account_insights_real():
    """Test getting insights with real Facebook credentials"""
    response = client.get(
        f"/facebook/account/{FB_ACCOUNT_ID}/insights?timeframe=7d",
        headers={"X-Facebook-Access-Token": FB_ACCESS_TOKEN}
    )
    assert response.status_code == 200
    data = response.json()
    assert "ads" in data
    assert "summary" in data
    assert isinstance(data["ads"], list)

@pytest.mark.skipif(
    not all([FB_ACCESS_TOKEN, FB_ACCOUNT_ID, FB_AD_ID]),
    reason="Facebook API credentials not configured"
)
def test_get_ad_demographics_real():
    """Test getting demographics with real Facebook credentials"""
    response = client.get(
        f"/facebook/ad/{FB_AD_ID}/demographics?timeframe=7d",
        headers={"X-Facebook-Access-Token": FB_ACCESS_TOKEN}
    )
    assert response.status_code == 200
    data = response.json()
    assert "spend_by_age" in data
    assert "spend_by_gender" in data
    assert "total_spend" in data


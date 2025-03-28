from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AdInsightModel(BaseModel):
    ad_id: str
    ad_name: str
    campaign_id: str
    campaign_name: str
    adset_id: str
    adset_name: str
    status: str # Note: This status comes from the filter, not directly from insights field
    platform: str = "Facebook"

    # Performance metrics
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    ctr: Optional[float] = None # Click-Through Rate (%)
    cpc: Optional[float] = None # Cost Per Click
    cpm: Optional[float] = None # Cost Per Mille

    # Conversion metrics
    conversions: int = 0 # Defaulting to 'purchase' action type count
    conversion_rate: Optional[float] = None
    cpa: Optional[float] = None # Cost Per Acquisition (for 'purchase')
    roas: Optional[float] = None # Return on Ad Spend (for 'purchase')

    # Engagement metrics
    video_views_3s: int = 0 # Approximation using video_play_actions or similar
    video_p75_watched: int = 0
    hook_rate: Optional[float] = None # (video_views_3s / impressions)
    hold_rate: Optional[float] = None # (video_p75_watched / video_views_3s)
    video_completion_rate: Optional[float] = None # (video_p100_watched / video_views_3s)
    comments: int = 0
    inline_link_clicks: int = 0
    inline_post_engagement: int = 0

    # Additional data
    thumbnail_url: Optional[str] = None
    embed_url: Optional[str] = None
    created_time: Optional[str] = None
    days_running: Optional[int] = None
    fetch_date: str = Field(default_factory=lambda: datetime.now().isoformat())


class FacebookInsightsResponseModel(BaseModel):
    account_id: str
    timeframe: str
    date_preset: str
    fetch_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    fetch_duration_seconds: float = 0.0
    ads: List[AdInsightModel] = []
    summary: Dict[str, Any] = {}


class DemographicDataModel(BaseModel):
    spend_by_age: Dict[str, float] = {}
    spend_by_gender: Dict[str, float] = {}
    total_spend: float = 0.0

class ThumbnailUrlModel(BaseModel):
    ad_id: str
    thumbnail_url: Optional[str] = None

class EmbedUrlModel(BaseModel):
    ad_id: str
    embed_url: Optional[str] = None

class ImageUrlModel(BaseModel):
    ad_id: str
    image_url: Optional[str] = None

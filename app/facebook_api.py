import logging
import time
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, APIRouter, HTTPException, Query, Path, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from facebook_business import FacebookAdsApi
from facebook_business.exceptions import FacebookRequestError
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adreportrun import AdReportRun
from facebook_business.adobjects.adsinsights import AdsInsights
from facebook_business.adobjects.ad import Ad
from facebook_business.adobjects.adcreative import AdCreative
from facebook_business.adobjects.advideo import AdVideo

from models import (
    AdInsightModel,
    FacebookInsightsResponseModel,
    DemographicDataModel,
    ThumbnailUrlModel,
    EmbedUrlModel,
    ImageUrlModel
)


router = APIRouter(prefix="/facebook", tags=["Facebook API"])


# --- Logging Configuration ---

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers(): # Avoid adding multiple handlers during reloads
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Constants ---

# Mapping of timeframe strings to Facebook date presets
TIMEFRAME_MAPPING = {
    "1d": "yesterday",
    "3d": "last_3d",
    "7d": "last_7d",
    "14d": "last_14d",
    "30d": "last_30d",
    "90d": "last_90d"
}

# Fields to fetch for insights
INSIGHTS_FIELDS = [
    AdsInsights.Field.impressions,
    AdsInsights.Field.clicks,
    AdsInsights.Field.spend,
    AdsInsights.Field.ctr, # Click-Through Rate
    AdsInsights.Field.cpc, # Cost Per Click (often calculated, but can be requested)
    AdsInsights.Field.cpm, # Cost Per Mille (1000 impressions)
    AdsInsights.Field.campaign_id,
    AdsInsights.Field.campaign_name,
    AdsInsights.Field.ad_id,
    AdsInsights.Field.ad_name,
    AdsInsights.Field.adset_id,
    AdsInsights.Field.adset_name,
    # Conversion metrics (requires correct pixel/event setup)
    AdsInsights.Field.actions, # Contains various actions like purchases, leads, etc.
    AdsInsights.Field.action_values, # Contains the value associated with actions
    AdsInsights.Field.cost_per_action_type, # CPA for specific actions
    AdsInsights.Field.website_purchase_roas, # Return On Ad Spend for website purchases
    # Video metrics
    AdsInsights.Field.video_play_actions, # General video plays (might include auto-plays)
    AdsInsights.Field.video_p25_watched_actions, # Watched 25%
    AdsInsights.Field.video_p50_watched_actions, # Watched 50%
    AdsInsights.Field.video_p75_watched_actions, # Watched 75%
    AdsInsights.Field.video_p100_watched_actions,# Watched 100% (Completion)
    AdsInsights.Field.video_30_sec_watched_actions, # Watched for 30 seconds
    AdsInsights.Field.video_avg_time_watched_actions, # Average watch time
    # Engagement metrics
    AdsInsights.Field.inline_link_clicks, # Clicks on links within the ad
    AdsInsights.Field.inline_post_engagement, # Likes, comments, shares, clicks
    # NOTE: 'comments' isn't a direct field, often inferred from 'actions' or 'inline_post_engagement'
    # We will parse 'actions' to find comment actions specifically.
]

# --- Data Models (using Pydantic for FastAPI integration) ---



# --- Facebook API Interaction Logic ---

def _init_api(access_token: str):
    """Initializes the Facebook Ads API with the provided token."""
    try:
        FacebookAdsApi.init(access_token=access_token, api_version='v18.0') # Use a specific API version
    except Exception as e:
        logger.error(f"Failed to initialize Facebook API: {e}")
        raise HTTPException(status_code=500, detail=f"Facebook API initialization failed: {e}")

# Dependency to get and validate the access token
async def get_access_token(x_facebook_access_token: str = Header(...)):
    if not x_facebook_access_token:
        raise HTTPException(status_code=400, detail="X-Facebook-Access-Token header is required")
    return x_facebook_access_token

# --- Insights Fetching Logic ---

def _parse_action_value(actions: List[Dict], action_type: str) -> int:
    """Safely extracts the value for a specific action type."""
    for action in actions:
        if action.get("action_type") == action_type:
            try:
                return int(float(action.get("value", 0)))
            except (ValueError, TypeError):
                return 0
    return 0

def _parse_action_field(insight: Dict, field_name: str, action_type: str) -> int:
    """Safely extracts action counts from fields like video_pXX_watched_actions."""
    actions = insight.get(field_name, [])
    return _parse_action_value(actions, action_type)

def _calculate_derived_metrics(ad_data: Dict) -> Dict:
    """Calculates derived metrics like CPC, CPA, Rates etc."""
    impressions = ad_data.get("impressions", 0)
    clicks = ad_data.get("clicks", 0)
    spend = ad_data.get("spend", 0.0)
    conversions = ad_data.get("conversions", 0)
    video_views_3s = ad_data.get("video_views_3s", 0)
    video_p75 = ad_data.get("video_p75_watched", 0)
    video_p100 = ad_data.get("video_p100_watched", 0) # Need p100 for completion rate

    ad_data["cpc"] = round(spend / clicks, 2) if clicks > 0 else 0.0
    ad_data["cpa"] = round(spend / conversions, 2) if conversions > 0 else 0.0
    ad_data["conversion_rate"] = round((conversions / clicks) * 100, 2) if clicks > 0 else 0.0
    ad_data["hook_rate"] = round((video_views_3s / impressions) * 100, 2) if impressions > 0 else 0.0
    ad_data["hold_rate"] = round((video_p75 / video_views_3s) * 100, 2) if video_views_3s > 0 else 0.0
    ad_data["video_completion_rate"] = round((video_p100 / video_views_3s) * 100, 2) if video_views_3s > 0 else 0.0

    # Facebook might return CTR/CPM directly, prefer those if available
    if "ctr" not in ad_data or ad_data["ctr"] is None:
         ad_data["ctr"] = round((clicks / impressions) * 100, 2) if impressions > 0 else 0.0
    if "cpm" not in ad_data or ad_data["cpm"] is None:
        ad_data["cpm"] = round((spend / impressions) * 1000, 2) if impressions > 0 else 0.0

    # Ensure ROAS is handled (might come directly as website_purchase_roas)
    if "roas" not in ad_data or ad_data["roas"] is None:
        # Calculate ROAS from action_values if not directly available
        purchase_values = [v for v in ad_data.get("action_values_raw", []) if v.get("action_type") == "purchase"]
        total_purchase_value = sum(float(v.get("value", 0)) for v in purchase_values)
        ad_data["roas"] = round(total_purchase_value / spend, 2) if spend > 0 else 0.0

    return ad_data

def _fetch_ad_details_batch(ad_ids: List[str], access_token: str) -> Dict[str, Dict]:
    """Fetches creation time, thumbnail, and embed URL for a batch of ads."""
    _init_api(access_token) # Ensure API is initialized for this thread/call
    details = {}
    batch_size = 50 # Facebook API limit

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for i in range(0, len(ad_ids), batch_size):
            batch_ids = ad_ids[i:i+batch_size]
            try:
                # Fetch creation time and creative ID in one go
                ads_data = Ad.get_by_ids(batch_ids, fields=[Ad.Field.id, Ad.Field.created_time, Ad.Field.creative])
                for ad_data in ads_data:
                    ad_id = ad_data[Ad.Field.id]
                    details[ad_id] = {
                        "created_time": ad_data.get(Ad.Field.created_time),
                        "creative_id": ad_data.get(Ad.Field.creative, {}).get('id')
                    }
            except FacebookRequestError as e:
                logger.error(f"Error fetching ad batch ({batch_ids}): {e}")
            except Exception as e:
                 logger.error(f"Unexpected error fetching ad batch ({batch_ids}): {e}")

        # Now fetch creative details based on creative_id (can be parallelized further)
        creative_ids_map = {ad_id: data['creative_id'] for ad_id, data in details.items() if data.get('creative_id')}
        creative_ids = list(creative_ids_map.values())

        if creative_ids:
            try:
                # Use futures for fetching thumbnail/embed URLs concurrently
                future_to_ad_id = {}
                for ad_id, creative_id in creative_ids_map.items():
                     # Submit tasks to fetch thumbnail and embed url
                     f_thumb = executor.submit(_get_ad_thumbnail_url_internal, creative_id, access_token) # Pass token
                     f_embed = executor.submit(_get_ad_embed_url_internal, creative_id, access_token) # Pass token
                     future_to_ad_id[f_thumb] = (ad_id, 'thumbnail_url')
                     future_to_ad_id[f_embed] = (ad_id, 'embed_url')

                for future in as_completed(future_to_ad_id):
                    ad_id, detail_type = future_to_ad_id[future]
                    try:
                        result = future.result()
                        if ad_id in details:
                            details[ad_id][detail_type] = result
                    except Exception as e:
                        logger.error(f"Error fetching {detail_type} for ad {ad_id}: {e}")

            except FacebookRequestError as e:
                 logger.error(f"Error fetching creative details batch: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error fetching creative details: {e}")

    return details


def fetch_facebook_insights_logic(
    account_id: str,
    timeframe: str,
    status_filter: str,
    access_token: str
) -> FacebookInsightsResponseModel:
    """Logic to fetch and process Facebook ad insights."""
    start_time = time.time()
    _init_api(access_token)

    if not account_id.startswith("act_"):
        account_id_fb = f"act_{account_id}"
    else:
        account_id_fb = account_id
        account_id = account_id.replace("act_", "") # Clean account_id for response model


    date_preset = TIMEFRAME_MAPPING.get(timeframe)
    if not date_preset:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Valid options: {list(TIMEFRAME_MAPPING.keys())}")

    response = FacebookInsightsResponseModel(
        account_id=account_id,
        timeframe=timeframe,
        date_preset=date_preset
    )

    status_map = {
        "active": ["ACTIVE"],
        "paused": ["PAUSED", "ADSET_PAUSED", "CAMPAIGN_PAUSED", "ARCHIVED", "PENDING_REVIEW"], # Broader paused definition
        "all": ["ACTIVE", "PAUSED", "ADSET_PAUSED", "CAMPAIGN_PAUSED", "ARCHIVED", "PENDING_REVIEW", "DISAPPROVED", "PREAPPROVED", "PENDING_BILLING_INFO", "CAMPAIGN_OFF", "ADSET_OFF", "WITH_ISSUES"] # More comprehensive 'all'
    }
    status_values = status_map.get(status_filter.lower())
    if status_values is None:
        raise HTTPException(status_code=400, detail="Invalid status_filter. Use 'active', 'paused', or 'all'.")

    try:
        ad_account = AdAccount(account_id_fb)
        params = {
            "level": "ad",
            "date_preset": date_preset,
            "limit": 1000, # Fetch more per page if possible
            "filtering": [
                {"field": "ad.effective_status", "operator": "IN", "value": status_values},
                {"field": "spend", "operator": "GREATER_THAN", "value": "0"} # Get ads with any spend
            ],
             "breakdowns": [] # No breakdowns needed for ad-level summary
        }

        logger.info(f"Requesting insights for {account_id_fb} with params: {params}")
        async_job = ad_account.get_insights(fields=INSIGHTS_FIELDS, params=params, is_async=True)

        while True:
            job_status = async_job.remote_read()
            status = job_status[AdReportRun.Field.async_status]
            percent = job_status[AdReportRun.Field.async_percent_completion]
            logger.info(f"Insights job status: {status}, Completion: {percent}%")
            if status == 'Job Completed':
                break
            elif status == 'Job Failed':
                logger.error(f"Insights job failed for account {account_id_fb}. Result: {job_status}")
                # Attempt to get error details if available
                try:
                    error_message = job_status.get('error', {}).get('message', 'Unknown error')
                    raise HTTPException(status_code=500, detail=f"Facebook insights job failed: {error_message}")
                except AttributeError:
                     raise HTTPException(status_code=500, detail="Facebook insights job failed.")
            elif status in ['Job Skipped', 'Job Cancelled']:
                 raise HTTPException(status_code=500, detail=f"Facebook insights job status: {status}")
            time.sleep(5) # Increase polling interval slightly

        raw_insights = list(async_job.get_result())
        logger.info(f"Successfully retrieved {len(raw_insights)} raw insight records.")

    except FacebookRequestError as e:
        logger.error(f"Facebook API error fetching insights for {account_id_fb}: {e}")
        raise HTTPException(status_code=e.api_error_code() or 500, detail=f"Facebook API Error: {e.api_error_message()}")
    except Exception as e:
        logger.error(f"Unexpected error fetching raw insights for {account_id_fb}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    # --- Process Insights ---
    processed_ads = []
    ad_ids = [insight.get("ad_id") for insight in raw_insights if insight.get("ad_id")]

    # Fetch additional ad details (creation time, thumbnail, embed) in parallel
    ad_details_map = {}
    if ad_ids:
        logger.info(f"Fetching details for {len(ad_ids)} ads...")
        ad_details_map = _fetch_ad_details_batch(ad_ids, access_token)
        logger.info(f"Finished fetching ad details.")


    for insight in raw_insights:
        ad_id = insight.get("ad_id")
        if not ad_id:
            continue

        ad_details = ad_details_map.get(ad_id, {})

        # Basic info
        ad_data = {
            "ad_id": ad_id,
            "ad_name": insight.get("ad_name", "N/A"),
            "campaign_id": insight.get("campaign_id", "N/A"),
            "campaign_name": insight.get("campaign_name", "N/A"),
            "adset_id": insight.get("adset_id", "N/A"),
            "adset_name": insight.get("adset_name", "N/A"),
            "status": status_filter, # Use the filter status as effective status isn't in fields
            "impressions": int(insight.get("impressions", 0)),
            "clicks": int(insight.get("clicks", 0)), # Usually means link clicks
            "spend": float(insight.get("spend", 0.0)),
            "ctr": float(insight.get("ctr", 0.0)) if insight.get("ctr") is not None else None,
            "cpc": float(insight.get("cpc", 0.0)) if insight.get("cpc") is not None else None,
            "cpm": float(insight.get("cpm", 0.0)) if insight.get("cpm") is not None else None,
            "inline_link_clicks": int(insight.get("inline_link_clicks", 0)),
            "inline_post_engagement": int(insight.get("inline_post_engagement", 0)),
            "thumbnail_url": ad_details.get("thumbnail_url"),
            "embed_url": ad_details.get("embed_url"),
            "created_time": ad_details.get("created_time"),
        }

        # Parse actions and action_values
        actions = insight.get("actions", [])
        action_values = insight.get("action_values", [])
        ad_data["action_values_raw"] = action_values # Keep raw values for potential ROAS calc

        ad_data["conversions"] = _parse_action_value(actions, "purchase") # Defaulting to purchase
        ad_data["comments"] = _parse_action_value(actions, "comment")

        # Parse video metrics (using specific action types within the fields)
        # Note: 'video_view' often represents 3s+ views. video_play_actions might be broader.
        ad_data["video_views_3s"] = _parse_action_field(insight, "video_play_actions", "video_view")
        ad_data["video_p75_watched"] = _parse_action_field(insight, "video_p75_watched_actions", "video_view")
        # Need P100 for completion rate calculation
        ad_data["video_p100_watched"] = _parse_action_field(insight, "video_p100_watched_actions", "video_view")


        # Parse ROAS if available directly
        roas_list = insight.get("website_purchase_roas", [])
        if roas_list:
            try:
                ad_data["roas"] = round(float(roas_list[0].get("value", 0.0)), 2)
            except (ValueError, TypeError, IndexError):
                ad_data["roas"] = None # Fallback to calculation if needed

        # Calculate days running
        if ad_data["created_time"]:
            try:
                created_dt = datetime.fromisoformat(ad_data["created_time"].replace('Z', '+00:00'))
                now_dt = datetime.now(created_dt.tzinfo) # Match timezone awareness
                ad_data["days_running"] = (now_dt - created_dt).days
            except ValueError:
                logger.warning(f"Could not parse created_time for ad {ad_id}: {ad_data['created_time']}")
                ad_data["days_running"] = None

        # Calculate derived metrics (CPC, CPA, Rates etc.)
        ad_data = _calculate_derived_metrics(ad_data)

        try:
             # Remove raw action values before creating the model
            ad_data.pop("action_values_raw", None)
            processed_ads.append(AdInsightModel(**ad_data))
        except Exception as e:
            logger.error(f"Error creating AdInsightModel for ad {ad_id}: {e} - Data: {ad_data}")
            # Optionally skip this ad or handle error differently

    response.ads = processed_ads
    response.summary = _calculate_summary_metrics(processed_ads) # Calculate summary
    response.fetch_duration_seconds = round(time.time() - start_time, 2)

    logger.info(f"Processed {len(response.ads)} ads for account {account_id}. Duration: {response.fetch_duration_seconds}s")
    return response

def _calculate_summary_metrics(ads: List[AdInsightModel]) -> Dict[str, Any]:
    """Calculate summary metrics across a list of AdInsightModel objects."""
    if not ads:
        return {}

    count = len(ads)
    total_spend = sum(ad.spend for ad in ads)
    total_impressions = sum(ad.impressions for ad in ads)
    total_clicks = sum(ad.clicks for ad in ads) # Assuming 'clicks' is the primary click metric
    total_conversions = sum(ad.conversions for ad in ads)
    total_video_views_3s = sum(ad.video_views_3s for ad in ads)
    total_video_p75 = sum(ad.video_p75_watched for ad in ads)
    total_video_p100 = sum(ad.video_completion_rate * ad.video_views_3s / 100 if ad.video_views_3s > 0 and ad.video_completion_rate is not None else 0 for ad in ads) # Approximate back total completions

    # Calculate weighted averages or overall rates where appropriate
    overall_ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions > 0 else 0.0
    overall_cpc = round(total_spend / total_clicks, 2) if total_clicks > 0 else 0.0
    overall_cpa = round(total_spend / total_conversions, 2) if total_conversions > 0 else 0.0
    # Note: Overall ROAS requires total revenue, which isn't directly summed here. Average ROAS is often used.
    # Average ROAS calculation needs care if some ads have 0 spend or 0 revenue.
    valid_roas_ads = [ad.roas for ad in ads if ad.roas is not None and ad.spend > 0]
    average_roas = round(sum(valid_roas_ads) / len(valid_roas_ads), 2) if valid_roas_ads else 0.0

    overall_hook_rate = round((total_video_views_3s / total_impressions) * 100, 2) if total_impressions > 0 else 0.0
    overall_hold_rate = round((total_video_p75 / total_video_views_3s) * 100, 2) if total_video_views_3s > 0 else 0.0
    overall_completion_rate = round((total_video_p100 / total_video_views_3s) * 100, 2) if total_video_views_3s > 0 else 0.0

    # Averages of ad-level metrics (less precise than overall but sometimes useful)
    average_ctr = round(sum(ad.ctr for ad in ads if ad.ctr is not None) / count, 2) if count > 0 else 0.0
    average_cpc = round(sum(ad.cpc for ad in ads if ad.cpc is not None and ad.cpc > 0) / len([ad for ad in ads if ad.cpc is not None and ad.cpc > 0]) if any(ad.cpc is not None and ad.cpc > 0 for ad in ads) else 0.0, 2)
    average_cpa = round(sum(ad.cpa for ad in ads if ad.cpa is not None and ad.cpa > 0) / len([ad for ad in ads if ad.cpa is not None and ad.cpa > 0]) if any(ad.cpa is not None and ad.cpa > 0 for ad in ads) else 0.0, 2)


    return {
        "ad_count": count,
        "total_spend": round(total_spend, 2),
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "total_video_views_3s": total_video_views_3s,
        "total_video_p75_watched": total_video_p75,
        "overall_ctr_percent": overall_ctr,
        "overall_cpc": overall_cpc,
        "overall_cpa": overall_cpa,
        "average_roas": average_roas, # Use average ROAS as overall is harder without total revenue
        "overall_hook_rate_percent": overall_hook_rate,
        "overall_hold_rate_percent": overall_hold_rate,
        "overall_video_completion_rate_percent": overall_completion_rate,
        # Include averages if needed, but overalls are usually better
        # "average_ctr": average_ctr,
        # "average_cpc": average_cpc,
        # "average_cpa": average_cpa,
    }


# --- Demographics Fetching Logic ---

def fetch_ad_demographics_logic(ad_id: str, timeframe: str, access_token: str) -> DemographicDataModel:
    """Logic to fetch demographic breakdown for a specific ad."""
    _init_api(access_token)
    date_preset = TIMEFRAME_MAPPING.get(timeframe)
    if not date_preset:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Valid options: {list(TIMEFRAME_MAPPING.keys())}")

    try:
        ad = Ad(ad_id)
        insights = ad.get_insights(
            fields=["spend"],
            params={
                "breakdowns": ["age", "gender"],
                "date_preset": date_preset,
                "limit": 500 # Usually enough breakdowns
            },
        )

        age_spend = {}
        gender_spend = {}
        total_spend = 0.0

        for insight in insights:
            spend = float(insight.get("spend", 0.0))
            total_spend += spend
            age = insight.get("age")
            gender = insight.get("gender")

            if age:
                age_spend[age] = round(age_spend.get(age, 0.0) + spend, 2)
            if gender:
                gender_spend[gender] = round(gender_spend.get(gender, 0.0) + spend, 2)

        return DemographicDataModel(
            spend_by_age=age_spend,
            spend_by_gender=gender_spend,
            total_spend=round(total_spend, 2),
        )

    except FacebookRequestError as e:
        logger.error(f"Facebook API error getting demographics for ad {ad_id}: {e}")
        raise HTTPException(status_code=e.api_error_code() or 500, detail=f"Facebook API Error: {e.api_error_message()}")
    except Exception as e:
        logger.error(f"Error getting demographics for ad {ad_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- Thumbnail Fetching Logic ---

def _get_ad_creative_details(creative_id: str, access_token: str) -> Optional[Dict]:
    """Helper to get specific fields from AdCreative."""
    # No need to init API here if called by another function that already did
    # _init_api(access_token) # Only if called independently
    try:
        creative = AdCreative(creative_id)
        # Request fields relevant for finding image/thumbnail URLs
        creative_details = creative.api_get(fields=[
            AdCreative.Field.id,
            AdCreative.Field.thumbnail_url, # Often for videos
            AdCreative.Field.image_url,     # Often for static images
            AdCreative.Field.image_hash,    # Can be used to construct URL
            AdCreative.Field.object_story_spec, # Contains link_data, video_data etc.
            AdCreative.Field.video_id,      # To potentially get video thumbnail
        ])
        return creative_details
    except FacebookRequestError as e:
        # Handle cases where creative might not exist or access is denied
        if e.api_error_code() == 100 and "creative" in e.api_error_message().lower():
             logger.warning(f"Creative ID {creative_id} not found or access denied.")
             return None
        logger.error(f"Facebook API error fetching creative {creative_id}: {e}")
        return None # Don't raise, just return None for this helper
    except Exception as e:
        logger.error(f"Unexpected error fetching creative {creative_id}: {e}")
        return None



def _get_ad_thumbnail_url_internal(ad_id_or_creative_id: str, access_token: str, return_image = False, is_creative_id: bool = False) -> Optional[str]:
    """Internal logic to find the thumbnail URL, accepts ad_id or creative_id."""
    _init_api(access_token) # Ensure API is initialized

    creative_details = None
    if is_creative_id:
        creative_details = _get_ad_creative_details(ad_id_or_creative_id, access_token)
    else:
        # Fetch creative ID from Ad ID first
        try:
            ad = Ad(ad_id_or_creative_id)
            ad_data = ad.api_get(fields=[Ad.Field.creative])
            creative_id = ad_data.get(Ad.Field.creative, {}).get('id')
            if creative_id:
                creative_details = _get_ad_creative_details(creative_id, access_token)
            else:
                 logger.warning(f"No creative ID found for ad {ad_id_or_creative_id}")
                 return None
        except FacebookRequestError as e:
            logger.error(f"Facebook API error getting creative ID for ad {ad_id_or_creative_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting creative ID for ad {ad_id_or_creative_id}: {e}")
            return None

    if not creative_details:
        logger.warning(f"Could not retrieve creative details for {'creative' if is_creative_id else 'ad'} {ad_id_or_creative_id}")
        return None

    # --- Logic to find the best available URL ---
    # 1. Direct thumbnail_url (common for videos)
    if creative_details.get('thumbnail_url') and not return_image:
        return creative_details['thumbnail_url']

    # 2. Direct image_url (common for static images)
    if creative_details.get('image_url') and return_image:
        return creative_details['image_url']
    
    # 2a. Check for image_url in video_data (nested structure)
    if creative_details.get('object_story_spec', {}).get('video_data', {}).get('image_url') and return_image:
        return creative_details['object_story_spec']['video_data']['image_url']

    # 3. Look inside object_story_spec (more complex structures)
    story_spec = creative_details.get('object_story_spec', {})
    if story_spec:
        # Check link_data (carousel, single link ads)
        link_data = story_spec.get('link_data')
        if link_data and link_data.get('picture'): # 'picture' is often the thumbnail field here
            return link_data['picture']
        if link_data and link_data.get('image_url'): # Sometimes it's image_url
             return link_data['image_url']

        # Check video_data
        video_data = story_spec.get('video_data')
        if video_data and video_data.get('image_url'): # Video thumbnail
            return video_data['image_url']

        # Check photo_data (less common for ads, more for page posts)
        photo_data = story_spec.get('photo_data')
        if photo_data and photo_data.get('url'):
            return photo_data['url'] # Or image_url if that exists

    # 4. Fallback using image_hash (construct URL - format might change)
    # This is less reliable as the URL structure isn't officially documented
    # if creative_details.get('image_hash'):
    #     logger.warning(f"Falling back to image_hash for creative {creative_details['id']}. URL might be unstable.")
    #     # This URL structure is an example and might not be correct or stable
    #     return f"https://lookaside.facebook.com/ads/image?asset_id={creative_details['image_hash']}&access_token={access_token}" # Placeholder example

    logger.warning(f"No thumbnail or image URL found for {'creative' if is_creative_id else 'ad'} {ad_id_or_creative_id}")
    return None

def fetch_ad_thumbnail_url_logic(ad_id: str, access_token: str) -> ThumbnailUrlModel:
    """Logic to get the thumbnail URL for a specific ad."""
    thumbnail_url = _get_ad_thumbnail_url_internal(ad_id, access_token, is_creative_id=False)
    return ThumbnailUrlModel(ad_id=ad_id, thumbnail_url=thumbnail_url)


# --- Embed URL Fetching Logic ---

def _get_ad_embed_url_internal(ad_id_or_creative_id: str, access_token: str, is_creative_id: bool = False) -> Optional[str]:
    """Internal logic to get Instagram embed URL from ad_id or creative_id."""
    _init_api(access_token)

    instagram_permalink = None
    creative_id = None

    try:
        if is_creative_id:
            creative_id = ad_id_or_creative_id
        else:
            ad = Ad(ad_id_or_creative_id)
            # Need creative ID first
            ad_data = ad.api_get(fields=[Ad.Field.creative])
            creative_id = ad_data.get(Ad.Field.creative, {}).get('id')

        if not creative_id:
            logger.warning(f"No creative ID found for ad {ad_id_or_creative_id}")
            return None

        # Fetch creative details focusing on instagram permalink
        creative = AdCreative(creative_id)
        creative_data = creative.api_get(fields=[AdCreative.Field.instagram_permalink_url])
        instagram_permalink = creative_data.get(AdCreative.Field.instagram_permalink_url)

        # Fallback: Sometimes needed via Ad object's preview link
        if not instagram_permalink or not instagram_permalink.startswith("https://www.instagram.com/p/"):
             if not is_creative_id: # Only possible if we started with ad_id
                 ad = Ad(ad_id_or_creative_id)
                 # Using preview_shareable_link can be less reliable for embed
                 preview_data = ad.api_get(fields=[Ad.Field.preview_shareable_link])
                 preview_link = preview_data.get("preview_shareable_link")
                 # This preview link isn't always an instagram post URL
                 # Parsing it might be fragile. Prefer instagram_permalink_url.
                 logger.warning(f"Using preview_shareable_link for ad {ad_id_or_creative_id} as fallback. May not work for embedding.")
                 # Attempt to parse if it looks like an instagram post
                 if preview_link and "instagram.com/p/" in preview_link:
                     instagram_permalink = preview_link


        if not instagram_permalink or not instagram_permalink.startswith("https://www.instagram.com/p/"):
            logger.warning(f"No suitable Instagram URL found for {'creative' if is_creative_id else 'ad'} {ad_id_or_creative_id}")
            return None

        # Extract shortcode and build embed URL
        # Example: https://www.instagram.com/p/Cabcdefg123/ -> shortcode = Cabcdefg123
        url_parts = instagram_permalink.strip('/').split('/')
        try:
            p_index = url_parts.index('p')
            if p_index + 1 < len(url_parts):
                shortcode = url_parts[p_index + 1]
                # Basic validation of shortcode format (alphanumeric, -, _)
                if shortcode and all(c.isalnum() or c in ['-', '_'] for c in shortcode):
                     return f"https://www.instagram.com/p/{shortcode}/embed"
                else:
                     logger.warning(f"Extracted shortcode '{shortcode}' looks invalid from URL: {instagram_permalink}")
                     return None
            else:
                 logger.warning(f"Could not find shortcode after '/p/' in URL: {instagram_permalink}")
                 return None
        except ValueError:
            logger.warning(f"Could not find '/p/' segment in Instagram URL: {instagram_permalink}")
            return None

    except FacebookRequestError as e:
        logger.error(f"Facebook API error getting embed URL for {'creative' if is_creative_id else 'ad'} {ad_id_or_creative_id}: {e}")
        return None # Don't raise, return None
    except Exception as e:
        logger.error(f"Error getting embed URL for {'creative' if is_creative_id else 'ad'} {ad_id_or_creative_id}: {e}")
        return None # Don't raise, return None


def fetch_ad_embed_url_logic(ad_id: str, access_token: str) -> EmbedUrlModel:
    """Logic to get the embed URL for a specific ad."""
    embed_url = _get_ad_embed_url_internal(ad_id, access_token, is_creative_id=False)
    return EmbedUrlModel(ad_id=ad_id, embed_url=embed_url)










@router.get(
    "/account/{account_id}/insights",
    response_model=FacebookInsightsResponseModel,
    summary="Fetch Ad Insights for an Account",
    description="Retrieves performance insights for ads within a specified Facebook ad account and timeframe.",
)
async def get_account_insights(
    account_id: str = Path(..., description="Facebook Ad Account ID (e.g., '123456789' or 'act_123456789')."),
    timeframe: str = Query("7d", description=f"Time period for data. Options: {list(TIMEFRAME_MAPPING.keys())}."),
    status_filter: str = Query("active", description="Filter ads by status: 'active', 'paused', or 'all'."),
    access_token: str = Depends(get_access_token), # Use dependency injection for token
):
    """
    Endpoint to fetch comprehensive ad insights.
    Requires 'X-Facebook-Access-Token' header.
    """
    try:
        result = fetch_facebook_insights_logic(account_id, timeframe, status_filter, access_token)
        return result
    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        logger.exception(f"Unhandled error in get_account_insights for {account_id}: {e}") # Log full traceback
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@router.get(
    "/ad/{ad_id}/demographics",
    response_model=DemographicDataModel,
    summary="Fetch Ad Demographic Breakdown",
    description="Retrieves spend breakdown by age and gender for a specific ad.",
)
async def get_single_ad_demographics(
    ad_id: str = Path(..., description="Facebook Ad ID (e.g., '987654321')."),
    timeframe: str = Query("7d", description=f"Time period for data. Options: {list(TIMEFRAME_MAPPING.keys())}."),
    access_token: str = Depends(get_access_token),
):
    """
    Endpoint to fetch demographic spend data for one ad.
    Requires 'X-Facebook-Access-Token' header.
    """
    try:
        result = fetch_ad_demographics_logic(ad_id, timeframe, access_token)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Unhandled error in get_single_ad_demographics for {ad_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@router.get(
    "/ad/{ad_id}/thumbnail",
    response_model=ThumbnailUrlModel,
    summary="Fetch Ad Thumbnail URL",
    description="Retrieves the thumbnail URL (or image URL) for a specific ad.",
)
async def get_single_ad_thumbnail(
    ad_id: str = Path(..., description="Facebook Ad ID (e.g., '987654321')."),
    access_token: str = Depends(get_access_token),
):
    """
    Endpoint to fetch the thumbnail URL for one ad.
    Requires 'X-Facebook-Access-Token' header.
    """
    try:
        result = fetch_ad_thumbnail_url_logic(ad_id, access_token)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Unhandled error in get_single_ad_thumbnail for {ad_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@router.get(
    "/ad/{ad_id}/embed_url",
    response_model=EmbedUrlModel,
    summary="Fetch Ad Instagram Embed URL",
    description="Retrieves an embeddable Instagram URL if the ad has an associated Instagram post.",
)
async def get_single_ad_embed_url(
    ad_id: str = Path(..., description="Facebook Ad ID (e.g., '987654321')."),
    access_token: str = Depends(get_access_token),
):
    """
    Endpoint to fetch the Instagram embed URL for one ad.
    Requires 'X-Facebook-Access-Token' header.
    """
    try:
        result = fetch_ad_embed_url_logic(ad_id, access_token)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Unhandled error in get_single_ad_embed_url for {ad_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# a route to return the image url
@router.get(
    "/ad/{ad_id}/image_url",
    response_model=ImageUrlModel,
    summary="Fetch Ad Image URL",
    description="Retrieves the direct image URL (rather than thumbnail) for a specific ad.",
)
async def get_single_ad_image_url(
    ad_id: str = Path(..., description="Facebook Ad ID (e.g., '987654321')."),
    access_token: str = Depends(get_access_token),
):
    """
    Endpoint to fetch the direct image URL for one ad.
    Requires 'X-Facebook-Access-Token' header.
    """
    # First get the creative details
    image_url = _get_ad_thumbnail_url_internal(ad_id, access_token, return_image=True, is_creative_id=False)
    return ImageUrlModel(
        ad_id=ad_id,
        image_url=image_url if image_url else None
    )


def get_ad_creative(ad_id, access_token=None):
    """Get the creative details for a specific ad"""
    if access_token:
        _init_api(access_token)  # Initialize API if token provided
        
    try:
        # Get the ad object
        ad = Ad(ad_id)
        # Fetch the ad details with the creative field
        ad_details = ad.api_get(fields=['creative'])
        
        if 'creative' not in ad_details or not ad_details['creative']:
            logger.warning(f"No creative found for ad ID: {ad_id}")
            return None
            
        creative_id = ad_details['creative']['id']
        
        # Get the creative details
        creative = AdCreative(creative_id)
        creative_details = creative.api_get(fields=[
            'object_story_spec',
            'video_id',
            'thumbnail_url',
            'effective_object_story_id',
            'asset_feed_spec',
            'image_url',
            'image_hash',
            'instagram_permalink_url'
        ])
        
        return creative_details
    except FacebookRequestError as e:
        logger.warning(f"Facebook API error fetching creative for ad {ad_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching ad creative for ad {ad_id}: {e}")
        return None

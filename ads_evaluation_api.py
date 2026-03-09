# pm2 start "python ads_evaluation_api.py" --name ads_evaluation_api


import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, urlunparse

import boto3
import uvicorn
from database_file import close_connections, get_session
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, desc, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

# FastAPI app instance
app = FastAPI(
    title="Ads Evaluation API",
    description="API for managing ad evaluation, feedback, and combined ad retrieval",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Database models (copied from original file)
# -----------------------------
class Base(DeclarativeBase):
    pass


class AdCentral(Base):
    __tablename__ = "gpt_ad_central_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer)
    response = Column(JSONB, nullable=False)
    payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class GPTBlueprint(Base):
    __tablename__ = "gpt_blueprint_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=False)
    response = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class GPTFirstTime(Base):
    __tablename__ = "gpt_first_time_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=False)
    response = Column(JSONB, nullable=False)
    payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class AdsEvaluation(Base):
    __tablename__ = "ads_evaluation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=True)
    identifier = Column(Text, nullable=True)
    reference_ad_url = Column(Text, nullable=True)
    generated_ad_url = Column(Text, nullable=True)
    asset_url = Column(Text, nullable=True)
    ad_generation_prompt = Column(Text, nullable=True)
    ad_action = Column(Text, nullable=True)
    ad_feedback = Column(Text, nullable=True)
    ad_score = Column(Text, nullable=True)
    asset_action = Column(Text, nullable=True)
    asset_feedback = Column(Text, nullable=True)
    ad_generation_prompt_action = Column(Text, nullable=True)
    ad_generation_prompt_feedback = Column(Text, nullable=True)
    user_name = Column(Text, nullable=True)
    timestamp = Column(
        Text, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


class HeygenAvatarProductVideoGenerationLogs(Base):
    __tablename__ = "heygen_avatar_v4_video_generation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Text, nullable=True)
    heygen_video_id = Column(Text, nullable=True)
    brand_id = Column(Text, nullable=True)
    payload = Column(JSONB, nullable=True)
    s3_uri = Column(Text, nullable=True)
    status = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    lambda_invoked = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    e2e = Column(Boolean, nullable=True)
    source_environment = Column(Text, nullable=True)


class CloneAds(Base):
    __tablename__ = "clone_ads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=True)
    final_video_url = Column(Text, nullable=True)
    source_environment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ProductVideoGenerationLogs(Base):
    __tablename__ = "product_video_generation_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class UgcLongformVideos(Base):
    __tablename__ = "ugc_longform_videos"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=True)
    status = Column(Text, nullable=True)
    e2e = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ProductVideoV2Logs(Base):
    __tablename__ = "product_video_v2_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=True)
    lambda_invoked = Column(Boolean, nullable=True)
    status_code = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class IndustryVertical(Base):
    __tablename__ = "industry_vertical"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)


class BrandOverview(Base):
    __tablename__ = "brand_overview"
    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, nullable=False)
    brand_industry_vertical_id = Column(Integer)


class Brand(Base):
    __tablename__ = "brands"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    locale_id = Column(Text)
    is_active = Column(Boolean, default=True)


class Locale(Base):
    __tablename__ = "locale"
    id = Column(Integer, primary_key=True, autoincrement=True)
    langCode = Column(Text, primary_key=True, nullable=False)
    name = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)


# -----------------------------
# Language and Industry Mappings
# -----------------------------
language_map = {
    "English": ["en-us", "en-au", "en-nz", "en-ca", "en-sg", "en-gb"],
    "Russian": ["ru-ru"],
    "Portuguese": ["pt-pt", "pt-br"],
    "Spanish": ["es-mx", "es-ar", "es-cl", "es-es"],
    "Arabic": ["ar-sa", "ar-ae"],
    "Malay": ["ms-my"],
    "Japanese": ["ja-jp"],
    "Hindi": ["hi-in"],
    "Korean": ["ko-kr"],
    "French": ["fr-fr"],
    "Chinese": ["zh-cn", "zh-tw"],
    "German": ["de-de"],
    "Tagalog": ["tl-ph"],
    "Bengali": ["bn-bd"],
    "Tamil": ["ta-id"],
    "Turkish": ["tr-tr"],
    "Polish": ["pl-pl"],
    "Hungarian": ["hu-hu"],
    "Italian": ["it-it"],
    "Romanian": ["ro-ro"],
    "Norwegian": ["no-no"],
    "Hebrew": ["he-il"],
    "Danish": ["da-dk"],
    "Thai": ["th-th"],
    "Vietnamese": ["vi-vn"],
}

industry_map = {
    "Automotive": 1,
    "Beauty & Personal Care": 2,
    "Consumer Packaged Goods (CPG)": 3,
    "Education & Online Learning": 4,
    "Fashion & Apparel": 5,
    "Financial Services": 6,
    "Food & Beverage": 7,
    "Gaming": 8,
    "Health & Wellness": 9,
    "Media & Entertainment": 10,
    "Real Estate": 11,
    "Retail & E-commerce": 12,
    "Technology & Software": 13,
    "Telecommunications": 14,
    "Travel & Hospitality": 15,
}


# -----------------------------
# Pydantic Models for API
# -----------------------------
class AdModel(BaseModel):
    """Model representing a single ad"""

    reference_ad: Optional[str] = None
    generated_ad: Optional[str] = None
    prompt: Optional[str] = None
    asset: Optional[str] = None
    id: Optional[str] = None
    brand_id: Optional[int] = None
    timestamp: Optional[str] = None
    generated_by: Optional[str] = None


class FeedbackModel(BaseModel):
    """Model representing feedback for an ad"""

    score: int = Field(default=100, ge=0, le=1000)
    generated: Dict[str, Any] = Field(
        default_factory=lambda: {"action": None, "comment": ""}
    )
    prompt: Dict[str, Any] = Field(
        default_factory=lambda: {"action": None, "comment": ""}
    )
    asset: Dict[str, Any] = Field(
        default_factory=lambda: {"action": None, "comment": ""}
    )


class FeedbackUploadRequest(BaseModel):
    """Model for uploading feedback"""

    brand_id: Optional[int] = None
    identifier: Optional[str] = None
    reference_ad_url: Optional[str] = None
    generated_ad_url: Optional[str] = None
    asset_url: Optional[str] = None
    ad_generation_prompt: Optional[str] = None
    ad_action: Optional[str] = None
    ad_feedback: Optional[str] = None
    ad_score: Optional[str] = None
    asset_action: Optional[str] = None
    asset_feedback: Optional[str] = None
    ad_generation_prompt_action: Optional[str] = None
    ad_generation_prompt_feedback: Optional[str] = None
    user_name: Optional[str] = None


class PaginationMetadata(BaseModel):
    """Pagination metadata model"""

    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool


class CombinedAdsResponse(BaseModel):
    """Response model for combined ads"""

    ads: List[AdModel]
    pagination: PaginationMetadata
    brand_count: int = Field(description="Number of brands found")
    existing_feedback: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FeedbackResponse(BaseModel):
    """Response model for feedback"""

    feedback: Dict[str, FeedbackModel]
    pagination: PaginationMetadata


class FeedbackUploadResponse(BaseModel):
    """Response model for feedback upload"""

    success: bool
    action: str
    message: str


# -----------------------------
# Analytics Models
# -----------------------------
class BrandAdTypeCounts(BaseModel):
    brand_id: Optional[int]
    brand_name: Optional[str]
    counts: Dict[str, int]
    total: int


class AdRow(BaseModel):
    brand_id: Optional[int]
    brand_name: Optional[str]
    ad_type: str
    timestamp: Optional[str]


class AdGenerationSummaryResponse(BaseModel):
    ad_types: List[str]
    totals: Dict[str, int]
    brand_counts: List[BrandAdTypeCounts]
    total_brands: int
    total_ads: int
    rows: Optional[List[AdRow]] = None


# -----------------------------
# Utility Functions (copied from original file)
# -----------------------------
def strip_query_params(url: str) -> str:
    """Remove query parameters from URL"""
    parsed = urlparse(url)
    cleaned = parsed._replace(query="", fragment="")
    return urlunparse(cleaned)


def s3_uri_to_url(s3_uri: str) -> str:
    """Convert S3 URI to URL"""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    return f"https://{bucket}.s3.amazonaws.com/{key}"


def generate_presigned_url(presigned_url: str) -> str:
    """Generate presigned URL for S3 objects"""
    s3_client = boto3.client("s3", region_name="us-east-1")
    parsed_url = urlparse(presigned_url)
    object_key = parsed_url.path.lstrip("/")
    bucket_name = "ai-editing-dev"
    return s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name, "Key": object_key}
    )


def extract_id_from_url(url: str) -> str | None:
    """Extract ID from URL"""
    if not url:
        return None
    try:
        filename = os.path.basename(url)
        return os.path.splitext(filename)[0]
    except Exception:
        return None


def parse_brand_ids(brand_id: Union[str, int]) -> List[int]:
    """Parse brand_id parameter into list of integers"""
    if isinstance(brand_id, str):
        if "," in brand_id:
            return [int(b.strip()) for b in brand_id.split(",")]
        else:
            return [int(brand_id)]
    elif isinstance(brand_id, int):
        return [brand_id]
    elif isinstance(brand_id, list):
        return [int(b) for b in brand_id]
    else:
        raise ValueError("brand_id must be int, str, or list of ints/strings")


def get_brands_by_filters(
    session, industry=None, country=None, start_date=None, end_date=None, limit=10
):
    """Get brand IDs based on industry, country, and date filters"""
    from sqlalchemy import func

    # Parse dates
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0
        )
    else:
        start_dt = datetime.now() - timedelta(days=10)

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )
    else:
        end_dt = datetime.now().replace(hour=23, minute=59, second=59)

    # Build base query
    base_query = (
        session.query(
            BrandOverview.brand_id,
            func.row_number().over(order_by=desc(BrandOverview.id)).label("rn"),
        )
        .join(Brand, BrandOverview.brand_id == Brand.id)
        .filter(Brand.created_at.between(start_dt, end_dt))
    )

    # Add industry filter
    if industry and industry != "All":
        industry_id = industry_map.get(industry)
        if industry_id:
            base_query = base_query.filter(
                BrandOverview.brand_industry_vertical_id == industry_id
            )

    # Add country filter
    if country:
        langCodes = language_map.get(country, ["en-us"])
        base_query = base_query.join(Locale, Brand.locale_id == Locale.id).filter(
            Locale.langCode.in_(langCodes)
        )

    # Apply limit and get results, ordered by creation date descending
    ranked = base_query.subquery("ranked")
    query = session.query(ranked.c.brand_id).filter(ranked.c.rn <= limit)

    # Get brand_ids with their creation dates for sorting
    brand_results = (
        session.query(Brand.id, Brand.created_at)
        .filter(Brand.id.in_([r.brand_id for r in query.all()]))
        .order_by(desc(Brand.created_at))
        .all()
    )

    brand_ids = [r.id for r in brand_results]
    return brand_ids


def create_pagination_metadata(
    page: int, page_size: int, total_count: int
) -> PaginationMetadata:
    """Create pagination metadata"""
    import math

    total_pages = math.ceil(total_count / page_size) if page_size > 0 else 0
    has_next = page < total_pages
    has_previous = page > 1

    return PaginationMetadata(
        page=page,
        page_size=page_size,
        total_count=total_count,
        total_pages=total_pages,
        has_next=has_next,
        has_previous=has_previous,
    )


# -----------------------------
# Core Business Logic Functions (copied and adapted from original file)
# -----------------------------
def get_combined_ads(
    session,
    brand_id,
    page: int = 1,
    page_size: int = 10,
    source_filter: str = None,
    start_date: str = None,
    end_date: str = None,
):
    """Fetch ads concurrently from GPTFirstTime, AdCentral, GPTBlueprint, and HeygenAvatarProductVideoGenerationLogs tables with optional source filtering and date filtering."""
    all_ads = []

    # Normalize brand_id list
    if isinstance(brand_id, str):
        brand_ids = [int(brand_id)]
    elif isinstance(brand_id, int):
        brand_ids = [brand_id]
    elif isinstance(brand_id, list):
        brand_ids = [int(b) for b in brand_id]
    else:
        raise ValueError("brand_id must be int, str, or list of ints/strings")

    def fetch_first_time():
        session, _ = get_session()
        query = session.query(
            GPTFirstTime.response,
            GPTFirstTime.brand_id,
            GPTFirstTime.payload,
            GPTFirstTime.created_at,
        )

        # For "All Ads" mode, filter by ad creation date instead of brand_ids
        if start_date or end_date:
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                    hour=0, minute=0, second=0
                )
            else:
                start_dt = datetime.now() - timedelta(days=5)

            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
            else:
                end_dt = datetime.now().replace(hour=23, minute=59, second=59)

            query = query.filter(GPTFirstTime.created_at.between(start_dt, end_dt))
        else:
            # For other modes, filter by brand_ids
            query = query.filter(GPTFirstTime.brand_id.in_(brand_ids))

        results = query.order_by(desc(GPTFirstTime.created_at)).all()
        session.close()
        ads = []

        for row in results:
            if row and row[0]:
                try:
                    # Handle both dict and list formats
                    if isinstance(row[0], list) and len(row[0]) > 0:
                        r = row[0][0]
                    elif isinstance(row[0], dict):
                        r = row[0]
                    else:
                        continue

                    if not r or not r.get("presigned_url"):
                        continue

                    # Get asset safely
                    asset = ""
                    if (
                        row[2]
                        and isinstance(row[2], dict)
                        and ("user_assets" in row[2] or "asset_url" in row[2])
                    ):
                        if "user_assets" in row[2]:
                            user_assets = row[2]["user_assets"]
                            if isinstance(user_assets, list) and len(user_assets) > 0:
                                asset = user_assets[0]
                        elif "asset_url" in row[2]:
                            asset = row[2]["asset_url"]

                    ads.append(
                        {
                            "reference_ad": r.get("reference_ad_url"),
                            "generated_ad": strip_query_params(r["presigned_url"]),
                            "prompt": r.get("ad_generation_prompt", ""),
                            "asset": asset,
                            "id": extract_id_from_url(r.get("s3_uri", "")),
                            "brand_id": row[1],
                            "timestamp": row[3].isoformat() if row[3] else None,
                            "generated_by": "onboarding_ads",
                        }
                    )
                except Exception:
                    continue
        return ads

    def fetch_ad_central():
        session, _ = get_session()
        query = session.query(
            AdCentral.payload["user_assets"].astext.label("user_assets"),
            AdCentral.response["reference_ad_url"].astext.label("ref"),
            AdCentral.response["presigned_url"].astext.label("gen"),
            AdCentral.response["ad_generation_prompt"].astext.label("prompt"),
            AdCentral.response["persona_asset_url"].astext.label("persona"),
            AdCentral.response["product_asset_url"].astext.label("product"),
            AdCentral.response["lifestyle_asset_url"].astext.label("life"),
            AdCentral.response["s3_uri"].astext.label("s3_uri"),
            AdCentral.brand_id,
            AdCentral.created_at,
        )

        # For "All Ads" mode, filter by ad creation date instead of brand_ids
        if start_date or end_date:
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                    hour=0, minute=0, second=0
                )
            else:
                start_dt = datetime.now() - timedelta(days=5)

            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
            else:
                end_dt = datetime.now().replace(hour=23, minute=59, second=59)

            query = query.filter(AdCentral.created_at.between(start_dt, end_dt))
        else:
            # For other modes, filter by brand_ids
            query = query.filter(AdCentral.brand_id.in_(brand_ids))

        results = query.order_by(desc(AdCentral.created_at)).all()
        session.close()
        ads = []
        for r in results:
            if not r.gen:
                continue

            asset = None
            try:
                asset = (
                    r.persona
                    or r.product
                    or r.life
                    or (
                        json.loads(r.user_assets)[0]
                        if getattr(r, "user_assets", None)
                        else None
                    )
                )
            except Exception:
                asset = None

            ads.append(
                {
                    "reference_ad": r.ref,
                    "generated_ad": strip_query_params(r.gen),
                    "prompt": r.prompt or "",
                    "asset": asset or "",
                    "id": extract_id_from_url(r.s3_uri),
                    "brand_id": r.brand_id,
                    "timestamp": r.created_at.isoformat() if r.created_at else None,
                    "generated_by": "ad_central",
                }
            )
        return ads

    def fetch_blueprint():
        session, _ = get_session()
        query = session.query(
            GPTBlueprint.response[0]["reference_ad_url"].astext.label("ref"),
            GPTBlueprint.response[0]["presigned_url"].astext.label("gen"),
            GPTBlueprint.response[0]["prompt"].astext.label("prompt"),
            GPTBlueprint.response[0]["chosen_asset_url"].astext.label("asset"),
            GPTBlueprint.response[0]["s3_uri"].astext.label("s3_uri"),
            GPTBlueprint.brand_id,
            GPTBlueprint.created_at,
        )

        # For "All Ads" mode, filter by ad creation date instead of brand_ids
        if start_date or end_date:
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                    hour=0, minute=0, second=0
                )
            else:
                start_dt = datetime.now() - timedelta(days=5)

            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
            else:
                end_dt = datetime.now().replace(hour=23, minute=59, second=59)

            query = query.filter(GPTBlueprint.created_at.between(start_dt, end_dt))
        else:
            # For other modes, filter by brand_ids
            query = query.filter(GPTBlueprint.brand_id.in_(brand_ids))

        results = query.order_by(desc(GPTBlueprint.created_at)).all()
        session.close()
        ads = []
        for r in results:
            if not r.gen:
                continue
            ads.append(
                {
                    "reference_ad": r.ref,
                    "generated_ad": r.gen,  # generate_presigned_url(r.gen),
                    "prompt": r.prompt or "",
                    "asset": r.asset,
                    "id": extract_id_from_url(r.s3_uri),
                    "brand_id": r.brand_id,
                    "timestamp": r.created_at.isoformat() if r.created_at else None,
                    "generated_by": "creative_brief",
                }
            )
        return ads

    def fetch_avatar_ads():
        session, _ = get_session()
        query = session.query(
            HeygenAvatarProductVideoGenerationLogs.s3_uri,
            HeygenAvatarProductVideoGenerationLogs.brand_id,
            HeygenAvatarProductVideoGenerationLogs.payload,
            HeygenAvatarProductVideoGenerationLogs.created_at,
        )

        # For "All Ads" mode, filter by ad creation date instead of brand_ids
        if start_date or end_date:
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                    hour=0, minute=0, second=0
                )
            else:
                start_dt = datetime.now() - timedelta(days=5)

            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
            else:
                end_dt = datetime.now().replace(hour=23, minute=59, second=59)

            query = query.filter(
                HeygenAvatarProductVideoGenerationLogs.created_at.between(
                    start_dt, end_dt
                ),
                HeygenAvatarProductVideoGenerationLogs.source_environment == "prod",
            )
        else:
            # For other modes, filter by brand_ids
            query = query.filter(
                HeygenAvatarProductVideoGenerationLogs.brand_id.in_(
                    [str(bid) for bid in brand_ids]
                ),
                HeygenAvatarProductVideoGenerationLogs.source_environment == "prod",
            )

        results = query.order_by(
            desc(HeygenAvatarProductVideoGenerationLogs.created_at)
        ).all()
        session.close()
        ads = []

        for row in results:
            if row and row[0]:  # Check if s3_uri exists
                try:
                    # Get asset and prompt from payload if available
                    asset = ""
                    prompt = ""
                    if row[2] and isinstance(row[2], dict):
                        # Extract asset from payload
                        asset = row[2].get("asset_url", "")
                        # Extract prompt from payload
                        prompt = row[2].get("prompt", "") or row[2].get(
                            "ad_generation_prompt", ""
                        )

                    # Generate presigned URL for the video
                    generated_ad_url = s3_uri_to_url(row[0]) if row[0] else ""

                    ads.append(
                        {
                            "reference_ad": None,  # Avatar ads typically don't have reference ads
                            "generated_ad": generated_ad_url,
                            "prompt": prompt,
                            "asset": asset,
                            "id": extract_id_from_url(row[0]),
                            "brand_id": int(row[1])
                            if row[1] and str(row[1]).isdigit()
                            else None,
                            "timestamp": row[3].isoformat() if row[3] else None,
                            "generated_by": "avatar_ads",
                        }
                    )
                except Exception:
                    continue
        return ads

    # Determine which functions to run based on source_filter
    functions_to_run = []

    if not source_filter or source_filter == "all":
        # Run all sources including avatar ads
        functions_to_run = [
            fetch_first_time,
            fetch_ad_central,
            fetch_blueprint,
            fetch_avatar_ads,
        ]
    elif source_filter == "onboarding_ads":
        functions_to_run = [fetch_first_time]
    elif source_filter == "ad_central":
        functions_to_run = [fetch_ad_central]
    elif source_filter == "creative_brief":
        functions_to_run = [fetch_blueprint]
    elif source_filter == "avatar_ads":
        functions_to_run = [fetch_avatar_ads]
    else:
        # Invalid source_filter, default to all
        functions_to_run = [
            fetch_first_time,
            fetch_ad_central,
            fetch_blueprint,
            fetch_avatar_ads,
        ]

    # Run selected functions concurrently
    with ThreadPoolExecutor(max_workers=len(functions_to_run)) as executor:
        results = list(executor.map(lambda fn: fn(), functions_to_run))

    # Flatten combined ads
    for batch in results:
        all_ads.extend(batch)

    # Sort ads by ad creation timestamp in descending order (newest first)
    # Note: Date filtering is now done at the database level in each fetch function
    all_ads.sort(
        key=lambda ad: datetime.fromisoformat(ad["timestamp"].replace("Z", "+00:00"))
        if ad.get("timestamp")
        else datetime.min,
        reverse=True,
    )

    # Apply pagination
    total_count = len(all_ads)
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_ads = all_ads[start_index:end_index]

    return paginated_ads, total_count


def get_existing_feedback(session, brand_id, page: int = 1, page_size: int = 10):
    """Get existing feedback from database"""
    if isinstance(brand_id, str):
        brand_ids = [int(brand_id)]
    elif isinstance(brand_id, int):
        brand_ids = [brand_id]
    elif isinstance(brand_id, list):
        brand_ids = [int(b) for b in brand_id]
    else:
        raise ValueError("brand_id must be int, str, or list of ints/strings")

    # First get the total count
    total_count = (
        session.query(AdsEvaluation)
        .filter(
            AdsEvaluation.brand_id.in_(brand_ids),
            AdsEvaluation.identifier.isnot(None),
        )
        .count()
    )

    # Then get the paginated results
    offset = (page - 1) * page_size
    rows = (
        session.query(
            AdsEvaluation.brand_id,
            AdsEvaluation.identifier,
            AdsEvaluation.ad_action,
            AdsEvaluation.ad_feedback,
            AdsEvaluation.ad_score,
            AdsEvaluation.asset_action,
            AdsEvaluation.asset_feedback,
            AdsEvaluation.ad_generation_prompt_action,
            AdsEvaluation.ad_generation_prompt_feedback,
        )
        .filter(
            AdsEvaluation.brand_id.in_(brand_ids),
            AdsEvaluation.identifier.isnot(None),
        )
        .offset(offset)
        .limit(page_size)
        .all()
    )

    feedback_map = {}
    for r in rows:
        key = f"{r.brand_id}:{r.identifier}"
        feedback_map[key] = {
            "score": int(r.ad_score)
            if r.ad_score and str(r.ad_score).isdigit()
            else 100,
            "generated": {"action": r.ad_action, "comment": r.ad_feedback or ""},
            "prompt": {
                "action": r.ad_generation_prompt_action,
                "comment": r.ad_generation_prompt_feedback or "",
            },
            "asset": {"action": r.asset_action, "comment": r.asset_feedback or ""},
        }
    return feedback_map, total_count


def upsert_ads_evaluation(data: dict):
    """
    Upsert an AdsEvaluation record based on the 'identifier' field.
    If the identifier exists → update specified fields.
    Else → insert new record.
    """
    session, _ = get_session()
    try:
        identifier = data.get("identifier")
        if not identifier:
            raise ValueError("Missing 'identifier' in data payload")

        # Try to find an existing record
        existing_record = (
            session.query(AdsEvaluation)
            .filter(AdsEvaluation.identifier == identifier)
            .first()
        )

        if existing_record:
            # Update only provided fields
            for key, value in data.items():
                if hasattr(existing_record, key) and value is not None:
                    setattr(existing_record, key, value)
            action = "updated"
        else:
            # Create new record
            new_record = AdsEvaluation(**data)
            session.add(new_record)
            action = "inserted"

        session.commit()
        return action
    finally:
        session.close()


# -----------------------------
# Analytics Helpers
# -----------------------------
def get_brand_names_map(session, brand_ids: List[int]) -> Dict[int, str]:
    """Return {brand_id: brand_name} for provided IDs."""
    if not brand_ids:
        return {}
    rows = session.query(Brand.id, Brand.name).filter(Brand.id.in_(brand_ids)).all()
    return {r.id: r.name for r in rows}


def map_source_to_ad_type(source: str) -> Optional[str]:
    """
    Map internal source to high-level dashboard ad type.
    - onboarding_ads, ad_central, creative_brief -> Image Ad
    - avatar_ads -> Avatar Video
    - Others → None (ignored)
    """
    if source in {"onboarding_ads", "ad_central", "creative_brief"}:
        return "Image Ad"
    if source == "avatar_ads":
        return "Avatar Video"
    return None


def _counts_grouped_by_brand(
    session, model, brand_col, created_at_col=None, extra_filters=None
):
    """
    Return {brand_id: count} using COUNT(*) GROUP BY brand_id.
    - extra_filters: list of SQLAlchemy filter expressions
    - created_at_col used only if provided along with date filters in caller
    """
    q = session.query(brand_col, func.count().label("cnt"))
    if extra_filters:
        for f in extra_filters:
            q = q.filter(f)
    q = q.group_by(brand_col)
    rows = q.all()
    result: Dict[Optional[int], int] = {}
    for bid, cnt in rows:
        # Normalize brand id to int where possible
        try:
            norm = int(bid) if bid is not None else None
        except Exception:
            norm = None
        if norm is not None:
            result[norm] = int(cnt)
    return result


# -----------------------------
# API Endpoints
# -----------------------------


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Ads Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "fetch_combined_ads": "/api/ads/combined?brand_id={id}&page=1&page_size=10&source_filter=all",
            "fetch_all_latest_brands": "/api/ads/combined?brands=true&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&page=1&page_size=10",
            "fetch_feedback": "/api/feedback?brand_id={id}&page=1&page_size=10",
            "upload_feedback": "/api/feedback/upload",
        },
        "pagination": {
            "default_page_size": 10,
            "max_page_size": "unlimited",
            "page_starts_from": 1,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/ads/combined", response_model=CombinedAdsResponse)
async def fetch_combined_ads(
    brand_id: Optional[Union[str, int]] = Query(
        None, description="Brand ID (single ID or comma-separated list)"
    ),
    industry: Optional[str] = Query(None, description="Industry filter"),
    country: Optional[str] = Query(None, description="Country filter"),
    start_date: Optional[str] = Query(
        None, description="Start date for brand creation filter (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(
        None, description="End date for brand creation filter (YYYY-MM-DD)"
    ),
    brands: Optional[bool] = Query(
        None, description="If True, fetch all active brands within date range"
    ),
    limit: Optional[int] = Query(10, description="Number of brands to fetch"),
    source_filter: Optional[str] = Query(
        None,
        description="Filter by ad source: 'onboarding_ads', 'ad_central', 'creative_brief', 'avatar_ads', or 'all'",
    ),
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    page_size: int = Query(10, ge=1, description="Number of items per page"),
):
    """
    Fetch combined ads with flexible filtering options.

    Args:
        brand_id: Single brand ID or comma-separated list of brand IDs (optional)
        industry: Industry filter (optional)
        country: Country filter (optional)
        start_date: Start date for brand creation filter in YYYY-MM-DD format (optional)
        end_date: End date for brand creation filter in YYYY-MM-DD format (optional)
        brands: If True, fetch all active brands within date range, otherwise default to 10 days (optional)
        limit: Number of brands to fetch when using industry/country filters (default 10)
        source_filter: Filter by ad source - 'onboarding_ads', 'ad_central', 'creative_brief', 'avatar_ads', or 'all' (optional)
        page: Page number (starts from 1)
        page_size: Number of items per page (default 10)

    Returns:
        CombinedAdsResponse: List of ads with pagination metadata and brand count
    """
    try:
        session, tunnel = get_session()

        try:
            # Handle different search modes
            if brand_id:
                # Direct brand ID search
                brand_ids = parse_brand_ids(brand_id)
            elif brands is True:
                # All Ads mode - we'll filter by date at the database level in each table
                # Pass empty list as brand_ids since filtering is done by date in fetch functions
                brand_ids = []

                # For "All Ads" mode, if no dates provided, default to last 5 days
                if not start_date and not end_date:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=5)).strftime(
                        "%Y-%m-%d"
                    )
            else:
                # Search by industry/country with date filtering
                if not industry and not country:
                    raise HTTPException(
                        status_code=400,
                        detail="Either brand_id, brands filter, or industry/country filters must be provided",
                    )

                brand_ids = get_brands_by_filters(
                    session=session,
                    industry=industry,
                    country=country,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                )

            # Only check for empty brand_ids if not in "All Ads" mode
            if not brand_ids and not brands:
                return CombinedAdsResponse(
                    ads=[],
                    pagination=create_pagination_metadata(page, page_size, 0),
                    brand_count=0,
                    existing_feedback={},
                )

            # Get ads and feedback
            ads, total_count = get_combined_ads(
                session, brand_ids, page, page_size, source_filter, start_date, end_date
            )

            # For "All Ads" mode, we'll get feedback after we have the ads and their brand_ids
            if brands and ads:
                # Extract brand_ids from the ads we found
                found_brand_ids = list(set(ad["brand_id"] for ad in ads))
                feedback, _ = get_existing_feedback(
                    session, found_brand_ids, page, page_size
                )
            elif brand_ids:
                # For other modes, use the original brand_ids
                feedback, _ = get_existing_feedback(session, brand_ids, page, page_size)
            else:
                feedback = {}

            pagination = create_pagination_metadata(page, page_size, total_count)

            # Calculate brand count
            if brands and ads:
                # For "All Ads" mode, count unique brands from the ads we found
                brand_count = len(set(ad["brand_id"] for ad in ads))
            elif brand_ids:
                # For other modes, use the original brand_ids count
                brand_count = len(brand_ids) if isinstance(brand_ids, list) else 1
            else:
                brand_count = 0

            return CombinedAdsResponse(
                ads=[AdModel(**ad) for ad in ads],
                pagination=pagination,
                brand_count=brand_count,
                existing_feedback=feedback,
            )

        finally:
            close_connections(session, tunnel)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid parameter format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching combined ads: {str(e)}"
        )


@app.get("/api/feedback", response_model=FeedbackResponse)
async def fetch_feedback(
    brand_id: Union[str, int] = Query(
        ..., description="Brand ID (single ID or comma-separated list)"
    ),
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    page_size: int = Query(10, ge=1, description="Number of items per page"),
):
    """
    Fetch existing feedback from ads_evaluation table.

    Args:
        brand_id: Single brand ID or comma-separated list of brand IDs
        page: Page number (starts from 1)
        page_size: Number of items per page (default 10)

    Returns:
        FeedbackResponse: Dictionary of feedback mapped by brand_id:identifier with pagination metadata
    """
    try:
        # Handle comma-separated brand IDs
        if isinstance(brand_id, str) and "," in brand_id:
            brand_ids = [int(b.strip()) for b in brand_id.split(",")]
        else:
            brand_ids = brand_id

        session, tunnel = get_session()
        try:
            feedback_map, total_count = get_existing_feedback(
                session, brand_ids, page, page_size
            )
            pagination = create_pagination_metadata(page, page_size, total_count)

            # Convert to FeedbackModel format
            formatted_feedback = {}
            for key, value in feedback_map.items():
                formatted_feedback[key] = FeedbackModel(**value)

            return FeedbackResponse(feedback=formatted_feedback, pagination=pagination)
        finally:
            close_connections(session, tunnel)

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid brand_id format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching feedback: {str(e)}"
        )


@app.post("/api/feedback/upload", response_model=FeedbackUploadResponse)
async def upload_feedback(
    feedback_data: FeedbackUploadRequest = Body(
        ..., description="Feedback data to upload"
    ),
):
    """
    Upload/upsert feedback to ads_evaluation table.

    Args:
        feedback_data: FeedbackUploadRequest containing all feedback fields

    Returns:
        FeedbackUploadResponse: Success status and action performed
    """
    try:
        # Convert Pydantic model to dict, excluding None values
        data = feedback_data.model_dump(exclude_none=True)

        if not data:
            raise HTTPException(status_code=400, detail="No data provided for upload")

        action = upsert_ads_evaluation(data)

        return FeedbackUploadResponse(
            success=True, action=action, message=f"Feedback {action} successfully"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading feedback: {str(e)}"
        )


# -----------------------------
# Analytics Endpoint
# -----------------------------
@app.get(
    "/api/analytics/ad_generation_summary", response_model=AdGenerationSummaryResponse
)
async def ad_generation_summary(
    brand_id: Optional[Union[str, int]] = Query(
        None, description="Brand ID (single ID or comma-separated list)"
    ),
    brands: Optional[bool] = Query(
        None, description="If True, aggregate across all brands within date range"
    ),
    start_date: Optional[str] = Query(
        None, description="Start date for ad creation filter (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(
        None, description="End date for ad creation filter (YYYY-MM-DD)"
    ),
    source_filter: Optional[str] = Query(
        "all",
        description="Filter by source: 'onboarding_ads', 'ad_central', 'creative_brief', 'avatar_ads', 'clone_video', 'product_animation', 'commercial_24s', or 'all'",
    ),
    include_rows: Optional[bool] = Query(
        False,
        description="If True, include individual rows (brand, ad_type, timestamp)",
    ),
    rows_limit: Optional[int] = Query(
        1000, ge=1, description="Max rows to include when include_rows=True"
    ),
):
    """
    Aggregate how many brands and which brands used which tools to generate what type of ads.
    For now, spend/tools are ignored. Ad types returned:
    - Image Ad (Onboarding + Ad Central + Creative Brief)
    - Clone Video (placeholder)
    - Avatar Video
    - Avatar + Product (placeholder)
    - Product Animation (placeholder)
    - 24s Commercial (placeholder)
    """
    try:
        session, tunnel = get_session()
        try:
            # Normalize brand scope and dates
            # Parse date range if provided (applies to both brand_id and brands=true)
            start_dt = (
                datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
                if start_date
                else None
            )
            end_dt = (
                datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
                if end_date
                else None
            )

            if brand_id:
                brand_ids = parse_brand_ids(brand_id)
            elif brands:
                brand_ids = []
                # Default to last 30 days if no dates provided
                if not start_dt and not end_dt:
                    end_dt = datetime.now().replace(hour=23, minute=59, second=59)
                    start_dt = (datetime.now() - timedelta(days=30)).replace(
                        hour=0, minute=0, second=0
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Provide brand_id or set brands=true with an optional date range",
                )

            # Default to last 30 days if no dates provided (applies to both brand_id and brands=true)
            if not start_dt and not end_dt:
                end_dt = datetime.now().replace(hour=23, minute=59, second=59)
                start_dt = (datetime.now() - timedelta(days=30)).replace(
                    hour=0, minute=0, second=0
                )

            # Prepare constants
            DASHBOARD_AD_TYPES = [
                "Image Ad",
                "Clone Video",
                "Avatar Video",
                "Avatar + Product",
                "Product Animation",
                "24s Commercial",
            ]

            # Initialize totals and per-brand counts
            totals = {t: 0 for t in DASHBOARD_AD_TYPES}
            brand_counts_index: Dict[int, Dict[str, int]] = {}

            # Build filters per-table
            def date_filters(col):
                fs = []
                if start_dt:
                    fs.append(col >= start_dt)
                if end_dt:
                    fs.append(col <= end_dt)
                return fs

            # Source: Onboarding (GPTFirstTime) -> Image Ad
            if source_filter in (None, "all", "onboarding_ads"):
                if brand_ids:
                    fs = [GPTFirstTime.brand_id.in_(brand_ids)]
                    if start_dt or end_dt:
                        fs += date_filters(GPTFirstTime.created_at)
                else:
                    fs = date_filters(GPTFirstTime.created_at)
                image_from_onboarding = _counts_grouped_by_brand(
                    session,
                    GPTFirstTime,
                    GPTFirstTime.brand_id,
                    created_at_col=GPTFirstTime.created_at,
                    extra_filters=fs,
                )
            else:
                image_from_onboarding = {}

            # Source: Ad Central -> Image Ad
            if source_filter in (None, "all", "ad_central"):
                if brand_ids:
                    fs = [AdCentral.brand_id.in_(brand_ids)]
                    if start_dt or end_dt:
                        fs += date_filters(AdCentral.created_at)
                else:
                    fs = date_filters(AdCentral.created_at)
                image_from_ad_central = _counts_grouped_by_brand(
                    session,
                    AdCentral,
                    AdCentral.brand_id,
                    created_at_col=AdCentral.created_at,
                    extra_filters=fs,
                )
            else:
                image_from_ad_central = {}

            # Source: Creative Brief (GPTBlueprint) -> Image Ad
            if source_filter in (None, "all", "creative_brief"):
                if brand_ids:
                    fs = [GPTBlueprint.brand_id.in_(brand_ids)]
                    if start_dt or end_dt:
                        fs += date_filters(GPTBlueprint.created_at)
                else:
                    fs = date_filters(GPTBlueprint.created_at)
                image_from_brief = _counts_grouped_by_brand(
                    session,
                    GPTBlueprint,
                    GPTBlueprint.brand_id,
                    created_at_col=GPTBlueprint.created_at,
                    extra_filters=fs,
                )
            else:
                image_from_brief = {}

            # Source: Avatar -> Avatar Video
            if source_filter in (None, "all", "avatar_ads"):
                if brand_ids:
                    fs = [
                        HeygenAvatarProductVideoGenerationLogs.brand_id.in_(
                            [str(b) for b in brand_ids]
                        ),
                        HeygenAvatarProductVideoGenerationLogs.source_environment
                        == "prod",
                        HeygenAvatarProductVideoGenerationLogs.status == 200,
                    ]
                    if start_dt or end_dt:
                        fs += date_filters(HeygenAvatarProductVideoGenerationLogs.created_at)
                else:
                    fs = date_filters(HeygenAvatarProductVideoGenerationLogs.created_at)
                    fs.append(HeygenAvatarProductVideoGenerationLogs.source_environment == "prod")
                    fs.append(HeygenAvatarProductVideoGenerationLogs.status == 200)
                avatar_counts = _counts_grouped_by_brand(
                    session,
                    HeygenAvatarProductVideoGenerationLogs,
                    HeygenAvatarProductVideoGenerationLogs.brand_id,
                    created_at_col=HeygenAvatarProductVideoGenerationLogs.created_at,
                    extra_filters=fs,
                )
            else:
                avatar_counts = {}

            # Source: Clone Ads -> Clone Video
            if source_filter in (None, "all", "clone_video"):
                if brand_ids:
                    fs = [
                        CloneAds.brand_id.in_(brand_ids),
                        CloneAds.source_environment == "prod",
                    ]
                    if start_dt or end_dt:
                        fs += date_filters(CloneAds.created_at)
                else:
                    fs = date_filters(CloneAds.created_at)
                    fs.append(CloneAds.source_environment == "prod")
                fs.append(CloneAds.final_video_url.isnot(None))
                clone_counts = _counts_grouped_by_brand(
                    session,
                    CloneAds,
                    CloneAds.brand_id,
                    created_at_col=CloneAds.created_at,
                    extra_filters=fs,
                )
            else:
                clone_counts = {}

            # Source: Product Video V2 -> Product Animation
            if source_filter in (None, "all", "product_animation"):
                if brand_ids:
                    fs = [
                        ProductVideoV2Logs.brand_id.in_(brand_ids),
                        ProductVideoV2Logs.lambda_invoked.is_(True),
                        ProductVideoV2Logs.status_code == 200,
                    ]
                    if start_dt or end_dt:
                        fs += date_filters(ProductVideoV2Logs.created_at)
                else:
                    fs = date_filters(ProductVideoV2Logs.created_at)
                    fs.append(ProductVideoV2Logs.lambda_invoked.is_(True))
                    fs.append(ProductVideoV2Logs.status_code == 200)
                product_anim_counts = _counts_grouped_by_brand(
                    session,
                    ProductVideoV2Logs,
                    ProductVideoV2Logs.brand_id,
                    created_at_col=ProductVideoV2Logs.created_at,
                    extra_filters=fs,
                )
            else:
                product_anim_counts = {}

            # Source: UGC Longform -> 24s Commercial
            if source_filter in (None, "all", "commercial_24s"):
                if brand_ids:
                    fs = [
                        UgcLongformVideos.brand_id.in_(brand_ids),
                        UgcLongformVideos.status == "completed",
                        UgcLongformVideos.e2e.is_(True),
                    ]
                    if start_dt or end_dt:
                        fs += date_filters(UgcLongformVideos.created_at)
                else:
                    fs = date_filters(UgcLongformVideos.created_at)
                    fs.append(UgcLongformVideos.status == "completed")
                    fs.append(UgcLongformVideos.e2e.is_(True))
                commercial_24s_counts = _counts_grouped_by_brand(
                    session,
                    UgcLongformVideos,
                    UgcLongformVideos.brand_id,
                    created_at_col=UgcLongformVideos.created_at,
                    extra_filters=fs,
                )
            else:
                commercial_24s_counts = {}

            # Combine into per-brand counts
            all_brand_ids = (
                set(image_from_onboarding.keys())
                | set(image_from_ad_central.keys())
                | set(image_from_brief.keys())
                | set(avatar_counts.keys())
                | set(clone_counts.keys())
                | set(product_anim_counts.keys())
                | set(commercial_24s_counts.keys())
            )
            for bid in all_brand_ids:
                brand_counts_index.setdefault(bid, {t: 0 for t in DASHBOARD_AD_TYPES})
                # Image Ad sum
                img_total = (
                    image_from_onboarding.get(bid, 0)
                    + image_from_ad_central.get(bid, 0)
                    + image_from_brief.get(bid, 0)
                )
                brand_counts_index[bid]["Image Ad"] = img_total
                # Avatar Video
                brand_counts_index[bid]["Avatar Video"] = avatar_counts.get(bid, 0)
                # Avatar + Product mirrors Avatar Video
                brand_counts_index[bid]["Avatar + Product"] = avatar_counts.get(bid, 0)
                # Clone Video
                brand_counts_index[bid]["Clone Video"] = clone_counts.get(bid, 0)
                # Product Animation
                brand_counts_index[bid]["Product Animation"] = product_anim_counts.get(
                    bid, 0
                )
                # 24s Commercial
                brand_counts_index[bid]["24s Commercial"] = commercial_24s_counts.get(
                    bid, 0
                )
                # No remaining placeholders

            # Totals
            for bid, counts in brand_counts_index.items():
                for k, v in counts.items():
                    totals[k] += v

            # Build brand names and response objects
            active_brand_ids = sorted(
                [bid for bid, c in brand_counts_index.items() if sum(c.values()) > 0]
            )
            brand_name_map = get_brand_names_map(session, active_brand_ids)
            brand_counts: List[BrandAdTypeCounts] = [
                BrandAdTypeCounts(
                    brand_id=bid,
                    brand_name=brand_name_map.get(bid),
                    counts=brand_counts_index[bid],
                    total=sum(brand_counts_index[bid].values()),
                )
                for bid in active_brand_ids
            ]

            total_brands = len(brand_counts)
            total_ads = sum(totals.values())

            # Optional rows (lightweight; we do not fetch heavy columns)
            rows: List[AdRow] = []
            if include_rows and rows_limit > 0:
                # Build minimal per-ad rows only by reconstructing from counts is not possible;
                # for a lightweight approach, skip rows unless specifically required with another query.
                rows = []

            return AdGenerationSummaryResponse(
                ad_types=DASHBOARD_AD_TYPES,
                totals=totals,
                brand_counts=brand_counts,
                total_brands=total_brands,
                total_ads=total_ads,
                rows=rows if include_rows else None,
            )
        finally:
            close_connections(session, tunnel)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating ad summary: {str(e)}"
        )


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# -----------------------------
# Run the application
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("ads_evaluation_api:app", host="0.0.0.0", port=5858, reload=True)

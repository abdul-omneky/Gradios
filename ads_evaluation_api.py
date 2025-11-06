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
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, desc
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
# Utility Functions (copied from original file)
# -----------------------------
def strip_query_params(url: str) -> str:
    """Remove query parameters from URL"""
    parsed = urlparse(url)
    cleaned = parsed._replace(query="", fragment="")
    return urlunparse(cleaned)


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


def get_latest_brands(session, start_date=None, end_date=None):
    """Get all the brands ids that are active and within given date range"""
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

    # Query active brands within date range, ordered by creation date descending
    results = (
        session.query(Brand.id)
        .filter(Brand.is_active == True, Brand.created_at.between(start_dt, end_dt))
        .order_by(desc(Brand.created_at))
        .all()
    )

    brands = [r.id for r in results]
    return brands


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
    session, brand_id, page: int = 1, page_size: int = 10, source_filter: str = None
):
    """Fetch ads concurrently from GPTFirstTime, AdCentral, and GPTBlueprint tables with optional source filtering."""
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
        results = (
            session.query(
                GPTFirstTime.response,
                GPTFirstTime.brand_id,
                GPTFirstTime.payload,
                GPTFirstTime.created_at,
            )
            .filter(GPTFirstTime.brand_id.in_(brand_ids))
            .order_by(desc(GPTFirstTime.created_at))
            .all()
        )
        session.close()
        ads = []
        for row in results:
            if row and row[0] and isinstance(row[0], list):
                r = row[0][0]
                if not r.get("presigned_url"):
                    continue
                ads.append(
                    {
                        "reference_ad": r.get("reference_ad_url"),
                        "generated_ad": strip_query_params(r["presigned_url"]),
                        "prompt": r.get("ad_generation_prompt", ""),
                        "asset": row[2]["user_assets"][0]
                        if len(row[2]["user_assets"]) > 0
                        else "",
                        "id": extract_id_from_url(r.get("s3_uri", "")),
                        "brand_id": row[1],
                        "timestamp": row[3].isoformat() if row[3] else None,
                        "generated_by": "onboarding_ads",
                    }
                )
        return ads

    def fetch_ad_central():
        session, _ = get_session()
        results = (
            session.query(
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
            .filter(AdCentral.brand_id.in_(brand_ids))
            .order_by(desc(AdCentral.created_at))
        )
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
        results = (
            session.query(
                GPTBlueprint.response[0]["reference_ad_url"].astext.label("ref"),
                GPTBlueprint.response[0]["presigned_url"].astext.label("gen"),
                GPTBlueprint.response[0]["reference_ad_analysis"].astext.label(
                    "prompt"
                ),
                GPTBlueprint.response[0]["chosen_asset_url"].astext.label("asset"),
                GPTBlueprint.response[0]["s3_uri"].astext.label("s3_uri"),
                GPTBlueprint.brand_id,
                GPTBlueprint.created_at,
            )
            .filter(GPTBlueprint.brand_id.in_(brand_ids))
            .order_by(desc(GPTBlueprint.created_at))
        )
        session.close()
        ads = []
        for r in results:
            if not r.gen:
                continue
            ads.append(
                {
                    "reference_ad": r.ref,
                    "generated_ad": generate_presigned_url(r.gen),
                    "prompt": r.prompt or "",
                    "asset": r.asset,
                    "id": extract_id_from_url(r.s3_uri),
                    "brand_id": r.brand_id,
                    "timestamp": r.created_at.isoformat() if r.created_at else None,
                    "generated_by": "creative_brief",
                }
            )
        return ads

    # Determine which functions to run based on source_filter
    functions_to_run = []

    if not source_filter or source_filter == "all":
        # Run all sources
        functions_to_run = [fetch_first_time, fetch_ad_central, fetch_blueprint]
    elif source_filter == "onboarding_ads":
        functions_to_run = [fetch_first_time]
    elif source_filter == "ad_central":
        functions_to_run = [fetch_ad_central]
    elif source_filter == "creative_brief":
        functions_to_run = [fetch_blueprint]
    else:
        # Invalid source_filter, default to all
        functions_to_run = [fetch_first_time, fetch_ad_central, fetch_blueprint]

    # Run selected functions concurrently
    with ThreadPoolExecutor(max_workers=len(functions_to_run)) as executor:
        results = list(executor.map(lambda fn: fn(), functions_to_run))

    # Flatten combined ads
    for batch in results:
        all_ads.extend(batch)

    # Get brand creation dates for sorting
    brand_dates = {}
    if all_ads:
        brand_ids_in_ads = list(set(ad["brand_id"] for ad in all_ads))
        brand_date_results = (
            session.query(Brand.id, Brand.created_at)
            .filter(Brand.id.in_(brand_ids_in_ads))
            .all()
        )
        brand_dates = {r.id: r.created_at for r in brand_date_results}

    # Sort ads by brand creation date in descending order
    all_ads.sort(
        key=lambda ad: brand_dates.get(ad["brand_id"], datetime.min), reverse=True
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
        description="Filter by ad source: 'onboarding_ads', 'ad_central', 'creative_brief', or 'all'",
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
        source_filter: Filter by ad source - 'onboarding_ads', 'ad_central', 'creative_brief', or 'all' (optional)
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
                # Latest brands filter - get active brands within date range
                brand_ids = get_latest_brands(
                    session=session, start_date=start_date, end_date=end_date
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

            if not brand_ids:
                return CombinedAdsResponse(
                    ads=[],
                    pagination=create_pagination_metadata(page, page_size, 0),
                    brand_count=0,
                    existing_feedback={},
                )

            # Get ads and feedback
            # Execute get_combined_ads and get_existing_feedback in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ads = executor.submit(
                    get_combined_ads, session, brand_ids, page, page_size, source_filter
                )
                future_feedback = executor.submit(
                    get_existing_feedback, session, brand_ids, page, page_size
                )
                ads, total_count = future_ads.result()
                feedback, _ = future_feedback.result()

            pagination = create_pagination_metadata(page, page_size, total_count)

            return CombinedAdsResponse(
                ads=[AdModel(**ad) for ad in ads],
                pagination=pagination,
                brand_count=len(brand_ids) if isinstance(brand_ids, list) else 1,
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

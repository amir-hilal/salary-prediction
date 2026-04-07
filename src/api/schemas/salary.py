from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experience_level": 2,
                "employment_type": 3,
                "remote_ratio": 100,
                "company_size": 1,
                "work_year": 2023,
                "job_family": 2,
                "location_region": 3,
                "is_us_company": 1,
            }
        }
    )

    experience_level: Annotated[
        int,
        Field(ge=0, le=3, description="0=Entry-level, 1=Mid-level, 2=Senior, 3=Executive"),
    ]
    employment_type: Annotated[
        int,
        Field(ge=0, le=3, description="0=Part-time, 1=Freelance, 2=Contract, 3=Full-time"),
    ]
    remote_ratio: Literal[0, 50, 100] = Field(
        description="Remote work percentage: 0 (on-site), 50 (hybrid), or 100 (fully remote)"
    )
    company_size: Annotated[
        int,
        Field(ge=0, le=2, description="0=Small, 1=Medium, 2=Large"),
    ]
    work_year: Annotated[
        int,
        Field(ge=2020, le=2030, description="Calendar year of the role"),
    ]
    job_family: Annotated[
        int,
        Field(
            ge=0,
            le=5,
            description=(
                "0=Other, 1=Analytics, 2=Data Science, "
                "3=Data Engineering, 4=ML/AI, 5=Leadership"
            ),
        ),
    ]
    location_region: Annotated[
        int,
        Field(
            ge=0,
            le=3,
            description="0=Rest of World, 1=Asia Pacific, 2=Europe, 3=North America",
        ),
    ]
    is_us_company: Literal[0, 1] = Field(
        description="1 if the company is US-based, 0 otherwise"
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_salary": 125000.0,
                "salary_range_low": 110000.0,
                "salary_range_high": 140000.0,
                "currency": "USD",
                "model_version": "20260407_142809",
                "prediction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            }
        }
    )

    predicted_salary: float
    salary_range_low: float
    salary_range_high: float
    currency: str
    model_version: str
    prediction_id: str


class ErrorResponse(BaseModel):
    detail: str
    code: str
